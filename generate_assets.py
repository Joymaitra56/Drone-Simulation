import os
import sys
import math

def setup_packages():
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("Pillow library not found. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
        print("Pillow installed successfully!")

setup_packages()
from PIL import Image, ImageDraw

def generate_grid_texture(filename="grid_pattern.png", tile_size=128):
    """Generates a repeating tile with a futuristic neon-cyan grid and soft glow."""
    print(f"Generating neon grid texture: {filename}...")
    
    # Create background: very dark metallic blue/black
    bg_color = (6, 12, 24)
    img = Image.new("RGB", (tile_size, tile_size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Neon cyan color settings
    cyan_core = (0, 240, 255)
    cyan_glow = (0, 120, 200)
    
    # Grid lines at the borders to make it repeat seamlessly
    glow_width = 4
    for offset in range(-glow_width, glow_width + 1):
        opacity_factor = 1.0 - (abs(offset) / (glow_width + 1))
        current_glow = tuple(int(c * opacity_factor) for c in cyan_glow)
        # horizontal lines (top and bottom)
        draw.line([(0, offset), (tile_size - 1, offset)], fill=current_glow)
        draw.line([(0, tile_size - 1 + offset), (tile_size - 1, tile_size - 1 + offset)], fill=current_glow)
        # vertical lines (left and right)
        draw.line([(offset, 0), (offset, tile_size - 1)], fill=current_glow)
        draw.line([(tile_size - 1 + offset, 0), (tile_size - 1 + offset, tile_size - 1)], fill=current_glow)
        
    # Draw core sharp neon lines
    draw.line([(0, 0), (tile_size - 1, 0)], fill=cyan_core, width=2)
    draw.line([(0, tile_size - 1), (tile_size - 1, tile_size - 1)], fill=cyan_core, width=2)
    draw.line([(0, 0), (0, tile_size - 1)], fill=cyan_core, width=2)
    draw.line([(tile_size - 1, 0), (tile_size - 1, tile_size - 1)], fill=cyan_core, width=2)
    
    img.save(filename)
    print(f"Successfully saved grid texture to {os.path.abspath(filename)}")


def generate_terrain_heightmap(filename="rough_terrain_heightmap.png", size=512):
    """Generates rough Martian/Lunar terrain, but flattens the terrain at landing spots."""
    print(f"Generating rough Martian/Lunar terrain heightmap: {filename}...")
    img = Image.new("L", (size, size), 0)
    pixels = img.load()
    
    # Landing pad positions and their flat radii
    # format: (x, y, radius)
    pads = [
        (-4.0, -4.0, 0.8),  # Home Landing Spot
        (-4.0,  4.0, 0.6),  # Spot 1
        ( 4.0,  4.0, 0.6),  # Spot 2
        ( 4.0, -4.0, 0.6),  # Spot 3
        ( 0.0,  2.0, 0.6),  # Spot 4
        ( 0.0, -2.0, 0.6)   # Spot 5
    ]
    
    # Crater positions to simulate a Moon/Mars surface
    # format: (x, y, radius, depth)
    craters = [
        (2.0, 0.0, 1.5, 60),
        (-2.0, 2.0, 1.2, 50),
        (-2.0, -1.0, 0.9, 40),
        (3.0, -2.0, 1.4, 55),
        (0.0, 4.0, 1.1, 45)
    ]
    
    for x in range(size):
        for y in range(size):
            # Map pixel space to real-world coordinates [-8.0, 8.0]
            rx = -8.0 + 16.0 * x / (size - 1)
            ry = -8.0 + 16.0 * y / (size - 1)
            
            # 1. Base fractal noise height (hills and valleys)
            h_val = 120.0  # Baseline height
            h_val += 60.0 * math.sin(0.4 * rx) * math.cos(0.4 * ry)
            h_val += 30.0 * math.sin(1.1 * rx + 0.5) * math.cos(0.9 * ry - 0.2)
            h_val += 15.0 * math.sin(2.5 * rx) * math.sin(2.8 * ry)
            h_val += 5.0 * math.sin(6.0 * rx) * math.cos(5.5 * ry)
            
            # 2. Subtract craters
            for cx, cy, cr, cd in craters:
                dist = math.sqrt((rx - cx)**2 + (ry - cy)**2)
                if dist < cr * 1.5:
                    if dist < cr:
                        # Bowl shape (quadratic depression)
                        h_val -= cd * (1.0 - (dist / cr)**2)
                    else:
                        # Raised crater rim
                        t = (dist - cr) / (cr * 0.5)
                        h_val += (cd * 0.4) * (1.0 - t)
            
            # Ensure height is clamped to [10, 240] before flattening
            h_val = max(10.0, min(240.0, h_val))
            
            # 3. Apply flattening at landing pads
            # We want pads to be perfectly flat at height 0 (real world z=0),
            # which maps to grayscale value 0.
            flatten_factor = 1.0
            for px, py, pr in pads:
                dist = math.sqrt((rx - px)**2 + (ry - py)**2)
                if dist < pr:
                    flatten_factor = 0.0
                    break
                elif dist < pr + 0.4:
                    # Smooth transition (slope) from flat (0.0) to full rough (1.0)
                    t = (dist - pr) / 0.4
                    # Smoothstep interpolation
                    factor = 3 * t**2 - 2 * t**3
                    if factor < flatten_factor:
                        flatten_factor = factor
            
            # Apply flatten factor
            final_h = h_val * flatten_factor
            pixels[x, y] = max(0, min(255, int(final_h)))
            
    img.save(filename)
    print(f"Successfully saved terrain heightmap to {os.path.abspath(filename)}")

if __name__ == "__main__":
    generate_grid_texture()
    generate_terrain_heightmap()
    print("Asset generation complete!")
