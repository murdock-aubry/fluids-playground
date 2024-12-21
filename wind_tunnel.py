import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_fluids import Stable2dObject
import imageio
import matplotlib.colors as mcolors
from numerics import create_rotated_ellipse_mask



def create_initial_conditions(sim):
    ngrid = sim.ngrid

    sim.velocity = np.zeros((2, ngrid, ngrid))  
    
    # Initialize dye field
    sim.dye = np.zeros((ngrid, ngrid))  # Assuming sim has a dye field for concentration

    return sim


def inject_dye(sim, jet_intensity, subset_fraction=0.8):
    """Inject dye at random locations along the strip sim.dye[:, 0]."""
    ngrid = sim.ngrid
    
    # Calculate how many points to select
    total_points = ngrid
    subset_size = int(total_points * subset_fraction)
    
    # Randomly select subset of indices along the strip
    indices = np.random.choice(total_points, size=subset_size, replace=False)
    
    # Inject dye at the selected locations along the strip
    sim.dye[indices, 0] += jet_intensity


def run_simulation():
    ngrid = 2 ** 6
    niters = 100
    dt = 0.1
    viscosity = 0.001
    diffusion = 0.0001
    
    # Define injection parameters
    jet_intensity = 3


    object_mask = create_rotated_ellipse_mask(ngrid, ngrid, -15)


    sim = Stable2dObject(
        ngrid=ngrid, 
        diffusion=diffusion, 
        viscosity=viscosity, 
        dt=dt,
        dye_diffusion=0.01,  # Add dye diffusion parameter
        object_mask = object_mask,
        force = 10.0,
    )
    sim = create_initial_conditions(sim)

    # Create two subplots for velocity and dye
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8), tight_layout = True)

    # Separate normalizations for velocity and dye
    vel_norm = mcolors.Normalize(vmin=0, vmax=20)  # Adjusted for velocity range
    dye_norm = mcolors.Normalize(vmin=0, vmax=100)  # Adjusted for dye range
    frames = []

    amplitude = ngrid // 8  # How far to move from center
    frequency = 0.1


    for _ in range(niters):

        sim.object_mask = create_rotated_ellipse_mask(ngrid, ngrid, 180 * np.sin(_ / 10))

        corners = [
            (
                ngrid//4,  # x position
                ngrid//4 + int(amplitude * np.cos(frequency * _))   # y position
            ),
            (
                ngrid//4,  # opposite phase
                ngrid//4 + int(amplitude * np.sin(frequency * _ + np.pi))
            )
        ]
        # Inject dye and update simulation
        inject_dye(sim, jet_intensity)
        
        velocity, dye = sim.forward()

        # Calculate velocity magnitude
        velocity_mag = np.sqrt(velocity[0]**2 + velocity[1]**2)

        # Clear both axes
        ax1.clear()
        ax2.clear()
        ax3.clear()

        # Plot velocity
        im1 = ax1.imshow(velocity_mag, cmap='viridis', norm=vel_norm)
        ax1.set_title('Velocity Magnitude')
        ax1.axis('off')

        # Plot dye
        im2 = ax2.imshow(dye, cmap='RdBu', norm=dye_norm)
        ax2.set_title('Dye Concentration')
        ax2.axis('off')
        ax2.imshow(sim.object_mask, cmap='binary', alpha=0.5)


        ax3.streamplot(np.arange(ngrid), np.arange(ngrid), sim.velocity[1], sim.velocity[0], color='r', linewidth=1)
        ax3.imshow(sim.object_mask, cmap='binary', alpha=0.5)
        ax3.axis('off')


        # Add colorbars on first frame
        if _ == 0:
            fig.colorbar(im1, ax=ax1, shrink=0.8, label='Velocity Magnitude')
            fig.colorbar(im2, ax=ax2, shrink=0.8, label='Dye Concentration')

        # Capture frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    # Save animation
    imageio.mimsave('gifs/wind_tunnel_dye8.gif', frames, duration=0.1, loop=0)
    plt.close(fig)

if __name__ == "__main__":
    run_simulation()