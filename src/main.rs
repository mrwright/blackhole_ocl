extern crate clap;
extern crate image;
extern crate ocl;
extern crate sdl2;

use ocl::enums::{ImageChannelDataType, ImageChannelOrder, MemObjectType};
use ocl::{Image, ProQue};
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;

// Since Schwarzschild black holes are spherically symmetric, there's really
// only one dimension that matters for rays, which is the angle between the
// ray and the line connecting the camera to the center of the black hole.
// So, precompute a list of "outcomes" for rays over a range of angles; each
// one tells us whether the ray falls into the black hole (and if so, at what
// angle around the black hole it does), or whether it escapes (and if so, at
// what angle). We do this with its own opencl kernel.
fn generate_outcomes_gpu(min: f32, max: f32, num: u32, start_r: f32) -> (Vec<f32>, Vec<u8>) {
    let src = include_str!("rays.ocl.c");

    let pro_que = ProQue::builder().src(src).dims(num).build().unwrap();
    let angle_buffer = pro_que.create_buffer::<f32>().unwrap();
    let outcome_buffer = pro_que.create_buffer::<u8>().unwrap();

    let kernel = pro_que
        .kernel_builder("gen_outcomes")
        .arg(&angle_buffer)
        .arg(&outcome_buffer)
        .arg(min)
        .arg(max)
        .arg(num)
        .arg(start_r)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut angle_vec = vec![0.0f32; angle_buffer.len()];
    let mut outcome_vec = vec![0u8; outcome_buffer.len()];
    angle_buffer.read(&mut angle_vec).enq().unwrap();
    outcome_buffer.read(&mut outcome_vec).enq().unwrap();

    (angle_vec, outcome_vec)
}

fn build_image(
    pro_que: &ProQue,
    data: &[u8],
    dims: (u32, u32),
) -> Result<ocl::Image<u8>, ocl::Error> {
    Image::<u8>::builder()
        .channel_order(ImageChannelOrder::Rgba)
        .channel_data_type(ImageChannelDataType::UnormInt8)
        .image_type(MemObjectType::Image2d)
        .dims(&dims)
        .flags(
            ocl::flags::MEM_READ_ONLY
                | ocl::flags::MEM_HOST_WRITE_ONLY
                | ocl::flags::MEM_COPY_HOST_PTR,
        )
        .copy_host_slice(data)
        .queue(pro_que.queue().clone())
        .build()
}

fn load_image(filename: &str, pro_que: &ProQue) -> Result<ocl::Image<u8>, String> {
    let img = image::open(filename)
        .map_err(|err| format!("Cannot open {}: {}", filename, err.to_string()))?
        .to_rgba();
    let dims = img.dimensions();

    Ok(build_image(pro_que, &img, dims)?)
}

fn black_image(pro_que: &ProQue) -> Result<ocl::Image<u8>, String> {
    Ok(build_image(pro_que, &[0, 0, 0, 0], (1, 1))?)
}

// Everything we need to keep track of.
struct Schwarz {
    // The OpenCL state
    pro_que: ProQue,
    // Angle buffer
    angles: ocl::Buffer<f32>,
    // Buffer to render into
    destbuf: ocl::Buffer<u8>,
    // "result" buffer--whether a ray falls in or escapes
    angle_result: ocl::Buffer<u8>,
    // Sky texture
    skytex: ocl::Image<u8>,
    // Event horizon texture
    spheretex: ocl::Image<u8>,
    // Antialias factor. Applies to each dimension--so the number of rays
    // per pixel is the *square* of this.
    aa: u32,
    // Length of the angles and angle_result buffers
    num_outcomes: u32,
}

impl Schwarz {
    fn new(
        aa: u32,
        num_outcomes: u32,
        x_res: u32,
        y_res: u32,
        skybox_file: &str,
        surface_file: Option<&str>,
    ) -> Result<Schwarz, String> {
        let src = include_str!("render.ocl.c");

        // TODO: dimensions should be configurable.
        let pro_que = ProQue::builder()
            .src(src)
            .dims((x_res, y_res))
            .build()
            .unwrap();

        let dest_buffer = pro_que
            .buffer_builder()
            .len(x_res * y_res * 4)
            .build()
            .unwrap();

        println!("Generating...");
        let angle_buf = pro_que
            .buffer_builder::<f32>()
            .len(num_outcomes)
            .build()
            .unwrap();
        let angle_result_buf = pro_que
            .buffer_builder::<u8>()
            .len(num_outcomes)
            .build()
            .unwrap();
        let (angles, outcomes) = generate_outcomes_gpu(0., 5., num_outcomes, 100.);
        angle_buf.write(&angles).enq().unwrap();
        angle_result_buf.write(&outcomes).enq().unwrap();
        println!("Done");

        println!("Loading textures...");
        let sky = load_image(skybox_file, &pro_que)?;
        let sphere = match surface_file {
            Some(f) => load_image(f, &pro_que)?,
            _ => black_image(&pro_que)?,
        };
        println!("Done");

        Ok(Schwarz {
            pro_que,
            destbuf: dest_buffer,
            angles: angle_buf,
            angle_result: angle_result_buf,
            skytex: sky,
            spheretex: sphere,
            num_outcomes,
            aa,
        })
    }

    pub fn render(&self, dest: &mut [u8], x_res: u32, y_res: u32, pitch: u32, cx: f32, cy: f32) {
        let kernel = self
            .pro_que
            .kernel_builder("schwarz")
            .arg(&self.destbuf)
            .arg(&self.angles)
            .arg(&self.angle_result)
            .arg(x_res)
            .arg(y_res)
            .arg(pitch)
            .arg(cx)
            .arg(cy)
            .arg(&self.skytex)
            .arg(&self.spheretex)
            .arg(self.aa)
            .arg(self.num_outcomes)
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        // Read into the SDL buffer.
        // I expect this isn't the "proper" way to do this (it looks like surface
        // access isn't the preferred way to use SDL in rust in general), but it
        // works well enough...
        self.destbuf.read(dest).enq().unwrap();
    }
}

fn parse_args<'a>() -> clap::ArgMatches<'a> {
    use clap::{App, Arg};

    App::new("blackhole_ocl")
        .about("Visualizes Schwarzschild black holes")
        .arg(Arg::with_name("width")
             .value_name("width")
             .long("width")
             .help("Width of the image to create")
             .takes_value(true))
        .arg(Arg::with_name("height")
             .value_name("height")
             .long("height")
             .help("Height of the image to create")
             .takes_value(true))
        .arg(Arg::with_name("antialias")
             .value_name("antialias")
             .visible_alias("aa")
             .long("antialias")
             .help("Antialiasing factor. Number of rays per pixel will be the square of this number.")
             .takes_value(true))
        .arg(Arg::with_name("sky_file")
             .value_name("filename")
             .long("sky_file")
             .help("Filename for the skybox")
             .takes_value(true)
             .required(true))
        .arg(Arg::with_name("surface_file")
             .value_name("filename")
             .long("surface_file")
             .help("Filename for the event horizon texture (defaults to solid black)")
             .takes_value(true))
        .arg(Arg::with_name("fps")
             .long("fps")
             .help("Periodically print frame rate")
        )
        .get_matches()
}

fn main() -> Result<(), String> {
    let matches = parse_args();

    let x_res = matches
        .value_of("width")
        .unwrap_or("1600")
        .parse::<u32>()
        .map_err(|e| e.to_string())?;
    let y_res = matches
        .value_of("height")
        .unwrap_or("1200")
        .parse::<u32>()
        .map_err(|e| e.to_string())?;
    let aa = matches
        .value_of("antialias")
        .unwrap_or("4")
        .parse::<u32>()
        .map_err(|e| e.to_string())?;
    let skybox_filename = matches.value_of("sky_file").unwrap();
    let surface_filename = matches.value_of("surface_file");
    let fps = matches.is_present("fps");

    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;

    let window = video_subsystem
        .window("Schwarzschild black hole visualizer", x_res, y_res)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;

    let mut event_pump = sdl_context.event_pump().unwrap();

    let pixel_format = window.surface(&event_pump)?.pixel_format_enum();
    if pixel_format != PixelFormatEnum::ARGB8888 && pixel_format != PixelFormatEnum::RGB888 {
        return Err(format!("Can only handle ARGB8888/RGB888 pixel format right now; got {:?}", pixel_format));
    }

    // TODO: the number of outcomes could be made configurable.
    let schwarz = Schwarz::new(aa, 8192, x_res, y_res, skybox_filename, surface_filename)?;

    let mut time = std::time::SystemTime::now();
    let mut frames = 0;

    // "Effective" mouse position. This is a smoothed version of the physical position,
    // since we don't want small mouse movements to cause a "jump"--it's better to smooth
    // out the motion.
    let mut mx = x_res as f32 / 2.;
    let mut my = y_res as f32 / 2.;
    // Current physical mouse position.
    let mut cmx = mx;
    let mut cmy = my;
    // Acceleration factor for mouse smoothing. Each frame, the effective position moves
    // that fraction of the way toward the physical position.
    let acc = 0.25;

    'running: loop {
        // Update effective mouse position
        mx = (1. - acc) * mx + acc * cmx;
        my = (1. - acc) * my + acc * cmy;

        // FPS counters are nice, so why not.
        frames += 1;
        if fps && frames == 100 {
            let duration = time.elapsed().unwrap();
            println!(
                "{} frames in {}ms = {} fps",
                frames,
                duration.as_millis(),
                (frames * 1000) as f32 / (duration.as_millis() as f32)
            );
            time = std::time::SystemTime::now();
            frames = 0;
        }

        {
            // New scope because .surface borrows event_pump.
            let mut surface = window.surface(&event_pump)?;
            let pitch = surface.pitch() / 4; // We want the pitch in pixels
            let (x_res, y_res) = surface.size();
            let pixels = surface.without_lock_mut().unwrap();

            schwarz.render(pixels, x_res, y_res, pitch, mx, my);
            surface.update_window().unwrap();
        }
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::MouseMotion { x, y, .. } => {
                    cmx = x as f32;
                    cmy = y as f32;
                }
                _ => {}
            }
        }
    }

    Ok(())
}
