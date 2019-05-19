typedef struct Pixel {
  unsigned char a;
  unsigned char r;
  unsigned char g;
  unsigned char b;
} Pixel;

__constant sampler_t sampler_const =
  CLK_NORMALIZED_COORDS_TRUE |
  CLK_ADDRESS_REPEAT |
  CLK_FILTER_LINEAR;

Pixel pixel_from_img(image2d_t img, float2 coords) {
  float4 vals = read_imagef(img, sampler_const, coords) * (float4)(255.);

  Pixel res = {
               (unsigned char)vals.w,
               (unsigned char)vals.x,
               (unsigned char)vals.y,
               (unsigned char)vals.z
  };
  return res;
}

__constant float max_r = 5.;

struct res_t {
  float angle;
  unsigned char outcome;
};

struct res_t lookup(__global float *angles, __global unsigned char *angle_results, float pos) {
  int posi = (int)(pos);
  float f = pos - posi;
  struct res_t res;

  res.outcome = angle_results[posi];
  if (res.outcome != angle_results[posi + 1]) {
    res.angle = angles[posi];
  } else {
    res.angle = (1.-f) * angles[posi] + f * angles[posi+1];
  }

  return res;
}

__kernel void schwarz(
                      __global Pixel *buffer,
                      __global float *angles,
                      __global unsigned char *angle_results,
                      unsigned int x_res,
                      unsigned int y_res,
                      unsigned int pitch,
                      float cx, // mouse location
                      float cy,
                      read_only image2d_t skytex,
                      read_only image2d_t spheretex,
                      int aa,
                      unsigned int num_outcomes
                      ) {
  int pixel_loc = get_global_id(0) + get_global_id(1) * pitch;
  int res_r = 0;
  int res_g = 0;
  int res_b = 0;

  for (int aa_x = 0; aa_x < aa; aa_x++) {
    for (int aa_y = 0; aa_y < aa; aa_y++) {
      float x = (float)(get_global_id(0));
      float y = (float)(get_global_id(1));

      x += (float)(aa_x) / (float)(aa);
      y += (float)(aa_y) / (float)(aa);

      float2 p = (float2)((x - (float)(x_res)/2.) / (float)(x_res/2),
                          (y - (float)(y_res)/2.) / (float)(x_res/2)); // Note: x_res here is not a typo.
                                                                       // We want square pixels.

      float r = length(p) * 3.;

      struct res_t lookup_res = lookup(angles, angle_results, r * (float)(num_outcomes) / max_r);
      float angle_out = lookup_res.angle;
      unsigned char res = lookup_res.outcome;

      float pixel_angle = atan2(p.y, p.x);
      // The xy-plane goes through the equator. x is the screen's x; z is the screen's y.
      float3 loc_rect = (float3)(cos(angle_out), sin(angle_out), 0.);
      loc_rect = (float3)(cos(pixel_angle) * loc_rect.x + sin(pixel_angle) * loc_rect.z,
                          loc_rect.y,
                          -sin(pixel_angle) * loc_rect.x + cos(pixel_angle) * loc_rect.z);

      float x_angle = cx / 200.;
      float y_angle = (cy - 600.) / 200.;

      loc_rect = (float3)(loc_rect.x,
                          cos(y_angle) * loc_rect.y + sin(y_angle) * loc_rect.z,
                          -sin(y_angle) * loc_rect.y + cos(y_angle) * loc_rect.z);

      loc_rect = (float3)(cos(x_angle) * loc_rect.x + sin(x_angle) * loc_rect.y,
                          -sin(x_angle) * loc_rect.x + cos(x_angle) * loc_rect.y,
                          loc_rect.z);

      float phi = acos(loc_rect.z) / M_PI;
      float theta = (atan2(loc_rect.y, loc_rect.x) + M_PI) / (2. * M_PI);

      Pixel pixel;
      if (res == 0) {
        pixel = pixel_from_img(spheretex, (float2)(theta, phi));
      } else {
        // Why -theta here and not in the other case?
        // Because we're seeing the "front" of the event horizon, but
        // the "back" of the skybox.
        pixel = pixel_from_img(skytex, (float2)(-theta, phi));
      }
      res_r += pixel.r;
      res_g += pixel.g;
      res_b += pixel.b;
    }
  }

  res_r /= (aa*aa);
  res_g /= (aa*aa);
  res_b /= (aa*aa);

  Pixel final_pixel = { res_b, res_g, res_r, 0 };
  buffer[pixel_loc] = final_pixel;
}
