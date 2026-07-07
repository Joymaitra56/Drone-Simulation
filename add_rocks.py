import os
import random
import math

def generate_stones():
    # Landing pad coordinates (avoidance zones)
    # format: (x, y, avoidance_radius)
    pads = [
        (-4.0, -4.0, 1.1),  # Home Landing Spot
        (-4.0,  4.0, 1.0),  # Spot 1
        ( 4.0,  4.0, 1.0),  # Spot 2
        ( 4.0, -4.0, 1.0),  # Spot 3
        ( 0.0,  2.0, 1.0),  # Spot 4
        ( 0.0, -2.0, 1.0)   # Spot 5
    ]
    
    # Existing hand-placed rocks coordinates to avoid direct heavy overlap
    existing_rocks = [
        (-2.5, 1.5, 0.6),
        (2.5, 1.5, 0.6),
        (2.0, -2.5, 0.7),
        (-2.0, -2.5, 0.6),
        (0.0, 4.5, 0.6),
        (-4.5, 0.0, 0.5),
        (4.5, 0.0, 0.5),
        (0.0, -4.5, 0.5)
    ]
    
    stones_xml = []
    
    # We want 50 stones in total: 10 large/tall ones, and 40 smaller ones
    stone_configs = []
    for i in range(10):
        stone_configs.append(('large', i + 1))
    for i in range(40):
        stone_configs.append(('small', i + 1))
        
    random.seed(42) # Seed for reproducibility
    
    generated_positions = []
    
    for idx, (size_type, stone_id) in enumerate(stone_configs):
        # Retry loop to find a valid location
        valid = False
        rx, ry = 0.0, 0.0
        while not valid:
            rx = random.uniform(-7.2, 7.2)
            ry = random.uniform(-7.2, 7.2)
            
            # Check landing pads
            valid = True
            for px, py, r in pads:
                dist = math.sqrt((rx - px)**2 + (ry - py)**2)
                if dist < r:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check existing rocks
            for ex, ey, r in existing_rocks:
                dist = math.sqrt((rx - ex)**2 + (ry - ey)**2)
                if dist < r + 0.3:
                    valid = False
                    break
            if not valid:
                continue
                
            # Check previously generated new stones
            for gx, gy, gr in generated_positions:
                dist = math.sqrt((rx - gx)**2 + (ry - gy)**2)
                if dist < gr + 0.2:
                    valid = False
                    break
        
        # Decide physical dimensions and visual details
        if size_type == 'large':
            # Large rocks: height 0.8m to 1.6m (geom size rz = 0.4 to 0.8)
            sx = random.uniform(0.35, 0.55)
            sy = random.uniform(0.35, 0.55)
            sz = random.uniform(0.40, 0.75) # height
            z_pos = sz * 0.7  # slightly embedded
            geom_type = random.choice(['ellipsoid', 'box'])
            radius_guard = max(sx, sy)
            generated_positions.append((rx, ry, radius_guard))
            
            # Add to XML list
            body_name = f"dyn_large_rock_{stone_id}"
            euler = f"{random.uniform(-25, 25):.1f} {random.uniform(-25, 25):.1f} {random.uniform(0, 360):.1f}"
            
            xml = f'        <!-- Large Rock {stone_id} -->\n'
            xml += f'        <body name="{body_name}" pos="{rx:.2f} {ry:.2f} {z_pos:.2f}" euler="{euler}">\n'
            xml += f'            <geom type="{geom_type}" size="{sx:.2f} {sy:.2f} {sz:.2f}" material="rock_mat" condim="3"/>\n'
            # 50% chance of a glowing core/crystal vein
            if random.random() > 0.5:
                cx = random.uniform(-sx*0.5, sx*0.5)
                cy = random.uniform(-sy*0.5, sy*0.5)
                cz = sz * 0.9
                c_size = random.uniform(0.04, 0.08)
                c_mat = random.choice(['glow_cyan', 'glow_magenta'])
                xml += f'            <geom type="sphere" size="{c_size:.3f}" pos="{cx:.2f} {cy:.2f} {cz:.2f}" material="{c_mat}"/>\n'
            xml += '        </body>\n'
            stones_xml.append(xml)
            
        else:
            # Small rocks: height 0.15m to 0.5m (geom size rz = 0.08 to 0.25)
            sx = random.uniform(0.12, 0.24)
            sy = random.uniform(0.12, 0.24)
            sz = random.uniform(0.08, 0.22)
            z_pos = sz * 0.6  # slightly embedded
            geom_type = 'ellipsoid'
            radius_guard = max(sx, sy)
            generated_positions.append((rx, ry, radius_guard))
            
            body_name = f"dyn_small_rock_{stone_id}"
            euler = f"{random.uniform(-35, 35):.1f} {random.uniform(-35, 35):.1f} {random.uniform(0, 360):.1f}"
            
            xml = f'        <!-- Small Rock {stone_id} -->\n'
            xml += f'        <body name="{body_name}" pos="{rx:.2f} {ry:.2f} {z_pos:.2f}" euler="{euler}">\n'
            xml += f'            <geom type="{geom_type}" size="{sx:.2f} {sy:.2f} {sz:.2f}" material="rock_mat" condim="3"/>\n'
            xml += '        </body>\n'
            stones_xml.append(xml)
            
    return "".join(stones_xml)

def update_scene_xml():
    xml_file = "scene.xml"
    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} not found!")
        return
        
    with open(xml_file, 'r') as f:
        content = f.read()
        
    new_stones_xml = generate_stones()
    
    start_tag = "        <!-- DYNAMIC STONES START -->\n"
    end_tag = "        <!-- DYNAMIC STONES END -->\n"
    
    # Check if dynamic stones block already exists
    if start_tag in content and end_tag in content:
        print("Updating existing dynamic stones block in scene.xml...")
        parts = content.split(start_tag)
        pre_content = parts[0]
        post_content = parts[1].split(end_tag)[1]
        new_content = pre_content + start_tag + new_stones_xml + end_tag + post_content
    else:
        print("Inserting new dynamic stones block before </worldbody> in scene.xml...")
        target = "    </worldbody>"
        if target in content:
            new_block = start_tag + new_stones_xml + end_tag + "    </worldbody>"
            new_content = content.replace(target, new_block)
        else:
            print("Error: Could not find </worldbody> in scene.xml!")
            return
            
    with open(xml_file, 'w') as f:
        f.write(new_content)
        
    print("scene.xml updated successfully with 50 stones (10 large, 40 small)!")

if __name__ == "__main__":
    update_scene_xml()
