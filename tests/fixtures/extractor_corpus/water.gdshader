shader_type spatial;
render_mode blend_mix, cull_disabled;

uniform vec4 albedo : source_color = vec4(1.0);
uniform sampler2D noise_tex;

const float PI2 = 6.28318;

varying vec3 world_pos;

struct Ray {
	vec3 origin;
	vec3 dir;
};

float wave(float x, float freq) {
	return sin(x * freq);
}

void vertex() {
	world_pos = VERTEX;
}

void fragment() {
	ALBEDO = albedo.rgb * wave(world_pos.x, PI2);
}
