import taichi as ti

ti.init(arch=ti.gpu)

res = 700
fbm_octaves = 8
warp_freq = 8.0
noise_mag = 2.0
start_freq = 4.0
noise_h = 0.7
noise_f = 2.0
noise_a = 1.5
mouse_pos=ti.Vector.field(2, dtype=float, shape=1)
_shape = ti.field(float, shape=(res, res))
_velocity = ti.Vector.field(2, float, shape=(res, res))
_noise = ti.field(float, shape=(res, res))
_displace_source = ti.field(float, shape=(res, res))
_dye = ti.field(float, shape=(res, res))
_dye_source = ti.field(float, shape=(res, res))
_edge = ti.field(float, shape=(res, res))
_framebuffer = ti.field(float, shape=(res, res))
_framebuffer2 = ti.field(float, shape=(res, res))

gradient = ti.field(float, shape=(4))

cursor = ti.field(float, shape=(2))
cursor[0] = 0.5
cursor[1] = 0.5

@ti.func
def fract(i):
    return i - ti.floor(i)

@ti.func
def lerp(l, r, frac):
    return l + frac * (r - l)

@ti.func
def dot(l, r):
    return l.dot(r)

@ti.func
def sample(field, P):
    return field[int(P)]

@ti.func
def clamp(v, vmin, vmax):
    return min(vmax, max(vmin, v))

@ti.func
def smoothstep(x, a=0, b=1):
    t = clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)

@ti.func
def bilerp(field, P):
    I = int(P)
    x = fract(P)
    y = 1 - x
    return (sample(field, I + ti.Vector([1,1])) * x.x * x.y +
            sample(field, I + ti.Vector([1,0])) * x.x * y.y +
            sample(field, I + ti.Vector([0,0])) * y.x * y.y +
            sample(field, I + ti.Vector([0,1])) * y.x * x.y)

# https://www.shadertoy.com/view/4sfGzS
@ti.func
def hash(p):
    p  = fract( p*0.3183099+.1 )
    p *= 17.0
    return fract( p.x*p.y*p.z*(p.x+p.y+p.z) )

# https://www.shadertoy.com/view/Xsl3Dl
@ti.func
def hash_3d(p):
    p = ti.Vector([
        p.dot(ti.Vector([127.1,311.7, 74.7])),
        p.dot(ti.Vector([269.5,183.3,246.1])),
        p.dot(ti.Vector([113.5,271.9,124.6]))
    ])
    return -1 + 2 * fract(ti.sin(p)*43758.5453123)

@ti.func
def quintic_interpolate(l, r, t):
    t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    return lerp(l, r, t)

@ti.func
def random_gradient(i, j):
    random_val = 2920 * ti.sin(i * 21942.0 + j * 171324.0 + 8912.0) * ti.cos(i * 23157.0 * j * 217832.0 + 9758.0)
    return ti.Vector([ti.cos(random_val), ti.sin(random_val)])

@ti.func
def dot_grid_gradient(ipx, ipy, fpx, fpy):
    grad = random_gradient(ipx, ipy)
    return grad.x * (fpx-ipx) + grad.y * (fpy-ipy) # dot(grad, frac)

@ti.func
def gradient_noise_3d(p):
    '''
    or perlin 3d?
    '''
    i = ti.floor(p)
    f = fract(p)
    
    u = f*f*(3.0-2.0*f)

    return lerp( lerp(  lerp(   dot( hash_3d( i + ti.Vector([0.0,0.0,0.0]) ), f - ti.Vector([0.0,0.0,0.0]) ), 
                                dot( hash_3d( i + ti.Vector([1.0,0.0,0.0]) ), f - ti.Vector([1.0,0.0,0.0]) ), u.x),
                        lerp(   dot( hash_3d( i + ti.Vector([0.0,1.0,0.0]) ), f - ti.Vector([0.0,1.0,0.0]) ), 
                                dot( hash_3d( i + ti.Vector([1.0,1.0,0.0]) ), f - ti.Vector([1.0,1.0,0.0]) ), u.x), u.y),
                 lerp(  lerp(   dot( hash_3d( i + ti.Vector([0.0,0.0,1.0]) ), f - ti.Vector([0.0,0.0,1.0]) ), 
                                dot( hash_3d( i + ti.Vector([1.0,0.0,1.0]) ), f - ti.Vector([1.0,0.0,1.0]) ), u.x),
                        lerp(   dot( hash_3d( i + ti.Vector([0.0,1.0,1.0]) ), f - ti.Vector([0.0,1.0,1.0]) ), 
                                dot( hash_3d( i + ti.Vector([1.0,1.0,1.0]) ), f - ti.Vector([1.0,1.0,1.0]) ), u.x), u.y), u.z )

@ti.func
def perlin(i, j):
    ipx = ti.floor(i)
    ipy = ti.floor(j)
    ll = dot_grid_gradient(ipx, ipy, i, j)
    lr = dot_grid_gradient(ipx + 1.0, ipy, i, j)
    ul = dot_grid_gradient(ipx, ipy + 1.0, i, j)
    ur = dot_grid_gradient(ipx + 1.0, ipy + 1.0, i, j)
    lerpxl = quintic_interpolate(ll, lr, fract(i))
    lerpxu = quintic_interpolate(ul, ur, fract(i))
    return quintic_interpolate(lerpxl, lerpxu, fract(j)) * 0.5 + 0.5

@ti.func
def perlin_fbm_3d(x, y, z, h, f, a):
    gain = 2 ** (-h)
    t = 0.0
    for _ in range(fbm_octaves):
        t += a * gradient_noise_3d(ti.Vector([f *x, f * y, f * z]))
        f *= 2.0
        a *= gain
    return t

#https://www.ryanjuckett.com/photoshop-blend-modes-in-hlsl/
@ti.func
def soft_light(x:ti.f32, y:ti.f32) ->ti.f32:
    result = 0.0
    if y <= 0.5:
        result = x - (1-2*y)*x*(1-x)
    else:
        d = ((16*x-12)*x+4)*x if x <= 0.25 else ti.sqrt(x)
        result = x + (2*y-1)*(d-x)
    return result


#FBM Perlin noise
@ti.kernel
def draw_perlin_fbm(offset_z:ti.f32, freq:ti.f32, h:ti.f32, f:ti.f32, a:ti.f32):
    for i, j in _noise:
        u = i / res
        v = j / res
        noise = perlin_fbm_3d(u * freq, v * freq, offset_z, h,f,a)
        _noise[i, j] = noise * 0.5 + 0.5

@ti.kernel
def slope(strength:ti.f32):
    for P in ti.grouped(_displace_source):
        n = _displace_source[P]
        nx = _displace_source[P-ti.Vector([2, 0])]
        ny = _displace_source[P-ti.Vector([0, 2])]
        _velocity[P] = ti.Vector([n-nx, n-ny]) * strength

@ti.kernel
def comp_multiply():
    for P in ti.grouped(_shape):
        _noise[P] = _shape[P] * _noise[P]

@ti.kernel
def comp_screen():
    for P in ti.grouped(_dye):
        s1 = _dye[P]
        s2 = _noise[P]
        _dye[P] = s1 + s2 - s1*s2

@ti.kernel
def comp_add1():
    for P in ti.grouped(_noise):
        _displace_source[P] = _noise[P] + _dye[P]

@ti.kernel
def comp_add2():
    for P in ti.grouped(_framebuffer):
        _framebuffer[P] = _framebuffer[P] + _framebuffer2[P]

@ti.kernel
def blur():
    for P in ti.grouped(_dye):
        local_val = 0.0
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = int(ti.Vector([i-1, j-1]))
                local_val += _dye[P+offset]
        avg = local_val / 9.0
        _dye_source[P] = avg

@ti.kernel
def displacement():
    for P in ti.grouped(_dye_source):
        vel = _velocity[P]
        offsetPixel = bilerp(_dye_source, P+vel)
        _dye[P] = offsetPixel

@ti.kernel
def render():
    for P in ti.grouped(_dye):
        grey = _dye[P]
        gradient_p = grey*4.0
        flr = ti.floor(gradient_p)
        fract_p = gradient_p - flr
        color = gradient[flr] * (1.0 - fract_p) + gradient[flr+1] * fract_p
        _framebuffer[P] = color

@ti.kernel
def shape_disk(disk_radius:ti.f32):
    center = ti.Vector([cursor[0], cursor[1]])
    for P in ti.grouped(_shape):
        normalized_p = P / res
        p2c = normalized_p - center
        radius = p2c.norm()
        interop = smoothstep(radius, disk_radius - 0.01, disk_radius + 0.01)
        _shape[P] = lerp(1.0, 0.0, interop)

@ti.kernel
def die1(opacity:ti.f32):
    for P in ti.grouped(_dye):
        _dye[P] = _dye[P] * opacity

@ti.kernel
def die2(opacity:ti.f32):
    for P in ti.grouped(_noise):
        _noise[P] = _noise[P] * opacity

@ti.kernel
def tint(r:ti.f32):
    for P in ti.grouped(_framebuffer2):
        _framebuffer2[P] = _edge[P] * r

@ti.kernel
def edge_detect():
    for P in ti.grouped(_dye):
        gx = 0.0
        gy = 0.0
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = int(ti.Vector([i-1, j-1]))
                sample = _dye[P+offset]
                gx += sample * (i-1)
                gy += sample * (j-1)
        _edge[P] = ti.abs(gx) + ti.abs(gy)

@ti.kernel
def init_gradient():
    gradient[0] = 0
    gradient[1] = 128.0 / 255.0
    gradient[2] = 192.0 / 255.0
    gradient[3] = 1.0

@ti.kernel
def update_cursor():
    cursor[0] += ti.random() * 0.02 - 0.01
    cursor[1] += ti.random() * 0.02 - 0.01
    cursor[0] = fract(cursor[0])
    cursor[1] = fract(cursor[1])

@ti.kernel
def fill_dye():
    for P in ti.grouped(_dye):
        _dye[P] = 0.0

@ti.kernel
def fill_shape():
    for P in ti.grouped(_shape):
        _shape[P] = 0.0   

@ti.kernel
def clear_buff():
    for P in ti.grouped(_framebuffer):
        _framebuffer[P]=0.0

gui = ti.GUI('Wander', (res, res))
fill_dye()
now = 0.0
init_gradient()
while gui.running:
    now += 0.01
    draw_perlin_fbm(now, start_freq, noise_h, noise_f, noise_a)
    comp_add1()
    fill_shape()
    #update_cursor()
    gui.get_event()
    mouse_pos=gui.get_cursor_pos()
    cursor[0]=mouse_pos[0]
    cursor[1]=mouse_pos[1]
    if(gui.is_pressed(ti.GUI.RMB)):
        fill_dye()
    if(gui.is_pressed(ti.GUI.LMB)):
        shape_disk(0.03)
    comp_multiply()
    die1(0.995)
    die2(0.3)
    comp_screen()
    blur()
    slope(16.0)
    displacement()
    edge_detect()
    render()
    tint(0.4)
    comp_add2()
    gui.set_image(_framebuffer)
    gui.show()
