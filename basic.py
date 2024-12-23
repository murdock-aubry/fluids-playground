import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from stable_fluids import Stable2dObject
import imageio
import matplotlib.colors as mcolors
from numerics import create_rotated_ellipse_mask



def create_initial_conditions(sim, force_mag = 10.0):
    ngrid = sim.ngrid

    sim.velocity = np.zeros((2, ngrid, ngrid))  
    sim.dye = np.zeros((ngrid, ngrid))


    # sim.velocity[ngrid//4: 3 * ngrid//4, nrgid // 4] = force_mag
    
    return sim


def inject_dye(sim, points, jet_intensity, jet_radius):

    ngrid = sim.ngrid
    x, y = np.meshgrid(np.arange(ngrid), np.arange(ngrid))

    for cx, cy in points:
        mask = (x - cx) ** 2 + (y - cy) ** 2 < jet_radius ** 2
        sim.dye[mask] += jet_intensity



def run_simulation():
    ngrid = 2 ** 6
    niters = 100
    dt = 0.05
    viscosity = 0.001
    diffusion = 0.00001
    
    # Define injection parameters
    jet_intensity = 0
    jet_radius = 0



    sim = Stable2dObject(
        ngrid=ngrid, 
        diffusion=diffusion, 
        viscosity=viscosity, 
        dt=dt,
        dye_diffusion=0.005,  # Add dye diffusion parameter
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

    points = [(ngrid // 4, ngrid//4), (3 * ngrid // 4, 3 * ngrid // 4)]
    

    for iframe in range(niters):

        # Inject dye and update simulation
        inject_dye(sim, points, jet_intensity, jet_radius)
        
        velocity, dye = sim.forward()



        

        # Calculate velocity magnitude
        velocity_mag = np.sqrt(velocity[0]**2 + velocity[1]**2)


        # Clear both plots
        ax1.clear()
        ax2.clear()

        # Plot velocity
        im1 = ax1.imshow(velocity_mag, cmap='viridis', norm=vel_norm)
        ax1.set_title('Velocity Magnitude')

        # Plot dye
        im2 = ax2.imshow(dye, cmap='RdPu', norm=dye_norm)
        ax2.set_title('Dye Concentration')


        # Add colorbars on first frame
        if iframe == 0:
            fig.colorbar(im1, ax=ax1, shrink=0.8, label='Velocity Magnitude')
            fig.colorbar(im2, ax=ax2, shrink=0.8, label='Dye Concentration')

        # Capture frame
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    # Save animation
    imageio.mimsave('gifs/simple_test.gif', frames, duration=0.1, loop=0)
    plt.close(fig)

if __name__ == "__main__":
    run_simulation()