# About
`blackhole_ocl` is an interactive visualizer for uncharged, non-rotating (Schwarzschild) black holes, showing the gravitational lensing effects. It's written in Rust, using SDL and OpenCL.

Here's the kind of image it produces:

![Screenshot, showing gravitational lensing with a distorted town in the background and a texture of earth in the foreground
](/example.jpg?raw=true)

The image above was generated using [this](https://hdrihaven.com/hdri/?h=greenwich_park) as the sky texture and the 8k earth day map from [here](https://www.solarsystemscope.com/textures/) as the surface texture.

Obviously, this isn't what you'd *actually* see if you were near a black hole; the purpose is to make it easier to understand how light is being lensed, rather than a faithful simulation of exactly what you'd see. So:
* You can texture the event horizon (in the example above, it has a texture of the earth). In real life the horizon would appear black, but texturing it lets you see where along the horizon photons would fall in. (Alternatively, it shows you what you'd see if there were a textured sphere just outside the event horizon, and gravitational redshift wasn't an effect.) Note how you can see the north and south poles at the same time in the screenshot above, thanks to the bending of light--that's the sort of thing that texturing the event horizon lets you visualize.
* The sky texture is rendered as if it's "at infinity". If you give it something like a star field the results will be pretty true to what you'd actually see; if you give it a scene on earth it won't quite be, if there are objects nearby.
# Running
`cargo run --release -- --help` will give you commandline options. You must provide a sky texture (see links above for one possible choice); you can optionally specify an event horizon texture, the resolution to render at, and the antialiasing factor.
# Caveats
* Very little validation is done when it comes to things like pixel formats; right now, if SDL gives it a pixel format it's not expecting, it'll just crash.
* Rust's image library doesn't seem to handle `.hdr` files particularly well--they end up really dark. You'll probably want to convert any such files to jpeg first.
* I've only tested this on my own graphics card (Nvidia GeForce GTX 1080). I don't know how well (or even if) it runs on other cards, but would be happy to hear peoples' experiences!
