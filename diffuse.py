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
    sim.dye = np.zeros((ngrid, ngrid))

    return sim


def inject_dye(sim, corners, jet_intensity, jet_radius):

    ngrid = sim.ngrid
    x, y = np.meshgrid(np.arange(ngrid), np.arange(ngrid))

    for cx, cy in corners:
        mask = (x - cx) ** 2 + (y - cy) ** 2 < jet_radius ** 2
        sim.dye[mask] += jet_intensity  # Continuously add dye in the corner regions



def run_simulation():
    ngrid = 2 ** 6
    niters = 200
    dt = 0.05
    viscosity = 0.001
    diffusion = 0.00001
    
    # Define injection parameters
    jet_intensity = 40
    jet_radius = 2

    object_mask = create_rotated_ellipse_mask(ngrid, ngrid, -15)
    
    sim = Stable2dObject(
        ngrid=ngrid, 
        diffusion=diffusion, 
        viscosity=viscosity, 
        dt=dt,
        dye_diffusion=0.005,  # Add dye diffusion parameter
        object = object_mask,
    )
    sim = create_initial_conditions(sim)

    # Create two subplots for velocity and dye
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Separate normalizations for velocity and dye
    vel_norm = mcolors.Normalize(vmin=0, vmax=20)  # Adjusted for velocity range
    dye_norm = mcolors.Normalize(vmin=0, vmax=100)  # Adjusted for dye range
    frames = []

    amplitude = ngrid // 6  # How far to move from center
    frequency = 0.1

    for _ in range(niters):
        corners = [
            (
                ngrid//4 + int(amplitude * np.sin(frequency * _)),  # x position
                ngrid//4 + int(amplitude * np.cos(frequency * _))   # y position
            ),
            (
                3*ngrid//4 + int(amplitude * np.sin(frequency * _ + np.pi)),
                3*ngrid//4 + int(amplitude * np.cos(frequency * _ + np.pi))
            )
        ]

        sim.object_mask = create_rotated_ellipse_mask(ngrid, ngrid, 180 * np.sin(_ / 10))

        # Inject dye and update simulation
        inject_dye(sim, corners, jet_intensity, jet_radius)
        
        velocity, dye = sim.forward()

        # Calculate velocity magnitude
        velocity_mag = np.sqrt(velocity[0]**2 + velocity[1]**2)

        # Clear both axes
        ax1.clear()
        ax2.clear()

        # Plot velocity
        im1 = ax1.imshow(velocity_mag, cmap='viridis', norm=vel_norm)
        ax1.set_title('Velocity Magnitude')

        # Plot dye
        im2 = ax2.imshow(dye, cmap='RdPu', norm=dye_norm)
        ax2.set_title('Dye Concentration')
        ax2.imshow(sim.object_mask, cmap='binary', alpha=0.5)

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
    imageio.mimsave('gifs/fluid_with_dye_spin.gif', frames, duration=0.1, loop=0)
    plt.close(fig)

if __name__ == "__main__":
    run_simulation()