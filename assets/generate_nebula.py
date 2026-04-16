import math

def generate_svg(filename, num_dots=30):
    width, height = 1000, 1000
    center_x, center_y = width / 2, height / 2
    
    # Golden angle in radians
    golden_angle = math.pi * (3 - math.sqrt(5))
    
    output = []
    output.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">')
    output.append(f'  <rect width="{width}" height="{height}" fill="#000000" />')
    
    # Max radius for the spiral spread
    max_spread = 380
    
    for n in range(num_dots):
        # Calculate angle and distance from center
        theta = n * golden_angle
        # The distance r is proportional to the square root of n
        r = max_spread * math.sqrt(n) / math.sqrt(num_dots - 1)
        
        x = center_x + r * math.cos(theta)
        y = center_y + r * math.sin(theta)
        
        # Calculate dot radius: central dots are larger, fading gracefully to the edges
        # We'll use a slightly parabolic curve or linear drop
        # min radius 6, max radius 45
        t = n / (num_dots - 1) # goes from 0 to 1
        dot_r = 45 * (1 - t)**0.8 + 6
        
        output.append(f'  <circle cx="{x:.2f}" cy="{y:.2f}" r="{dot_r:.2f}" fill="#FFFFFF" />')
    
    output.append('</svg>')
    
    with open(filename, 'w') as f:
        f.write('\n'.join(output))

if __name__ == "__main__":
    generate_svg('/Users/simonsun/github_project/MyNebula/assets/mynebula_balanced_b.svg', 35)
    print("SVG Generated.")
