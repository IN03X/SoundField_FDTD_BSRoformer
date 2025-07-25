import random
import time

import numpy as np


class FDTD2D_Slice:
    def __init__(self, skip: int) -> None:
        self.skip = skip
        self.simulator = FDTD2D()

    def __getitem__(self, index: int) -> dict:
        r"""Get a data.

        t: time_step
        h: height
        w: weight

        Returns:
            new_data: dict
        """
        
        data = self.simulator()
        
        bnd = data["bnd"]  # (h, w)
        u = data["u"][0 :: self.skip]  # (t, h, w)
        t = data["t"][0 :: self.skip]  # (t,)

        i = random.randint(0, u.shape[0] - 2)
        u0 = u[0]
        curr_u = u[i]
        next_u = u[i + 1]

        new_data = {
            "bnd": bnd,
            "u0": u0,
            "curr_u": curr_u,
            "next_u": next_u
        }

        return new_data

    def __len__(self) -> int:
        return 10000  # Number of steps in an epoch, 1getitem = 1len

class FDTD2D_at_T:
    def __init__(self, skip: int, duration:float, select_mode:str, select_num:int) -> None:
        self.skip = skip
        self.simulator = FDTD2D(duration=duration)
        self.select_mode = select_mode
        self.select_num = select_num
        self.i = 0

    def __getitem__(self, index: int) -> dict:
        r"""Get a data.

        t: time_step
        h: height
        w: weight

        Returns:
            new_data: dict
        """
        
        data = self.simulator()

        bnd = data["bnd"]  # (h, w)
        u = data["u"][0 :: self.skip]  # (t, h, w)
        t = data["t"][0 :: self.skip]  # (t,)
        u0 = u[0]
        if self.select_mode == "random":
            i = random.randint(0, u.shape[0] - 1)
        elif self.select_mode == "order":
            idxs = np.linspace(0, u.shape[0] - 1, self.select_num).astype(int)
            i = idxs[index]

        ut = u[i]
        ti = t[i]

        data = {
                "dataset_name": "FDTD_2D",
                "bnd": bnd[None, ...], # (1, h, w)
                "u0": u0[None, ...], # (1, h, w)
                "t": ti[None, ...], # (1,)
                "ut": ut[None, ...], # (1, h, w)
        }

        return data

    def __len__(self) -> int:
        if self.select_mode == "random":
            return 100 # Number of steps in an epoch, 1getitem = 1len
        elif self.select_mode == "order":
            return self.select_num
        
class FDTD2D_at_xy:
    def __init__(self, skip: int, duration:float, dx:float=0.1, dy:float=0.1, sampling_frequency:float=48000) -> None:
        self.skip = skip
        self.simulator = FDTD2D(duration=duration,dx=dx,dy=dy,sampling_frequency=sampling_frequency)
        self.i = 0
        self.dx = dx
        self.dy = dy


    def __getitem__(self, index: int) -> dict:
        r"""Get a data.

        t: time_step
        h: height
        w: weight

        Returns:
            new_data: dict
        """
        
        data = self.simulator()

        bnd = data["bnd"]  # (h, w)
        u = data["u"][0 :: self.skip]  # (t, h, w)
        t = data["t"][0 :: self.skip]  # (t,)
        u0 = u[0]

        bnd_pointscloud = np.argwhere(bnd == 1)  # (L_bnd, 2)
        bnd_set = set(map(tuple, bnd_pointscloud))  
        while True:
            i = random.randint(0, u.shape[1] - 1)  # i ∈ [0, height - 1]
            j = random.randint(0, u.shape[2] - 1)  # j ∈ [0, width - 1]
            if (i, j) not in bnd_set:
                break

        # x = self.dx * i
        # y = self.dy * j
        x = np.array(i).astype(np.float32) #()
        y = np.array(j).astype(np.float32)
        uxy = u[:,i,j] #(l,)

        data = {
                "dataset_name": "FDTD_2D_RIR",
                "bnd": bnd[None, ...], # (1, h, w)
                "u0": u0[None, ...], # (1, h, w)
                "x": x[None, ...], # (1,)
                "y": y[None, ...], # (1,)
                "uxy": uxy[..., None, None], # (l, 1, 1)
        }

        return data

    def __len__(self) -> int:
        return 100 # Number of steps in an epoch, 1getitem = 1len
    
class FDTD2D_at_xy_pointscloud:
    def __init__(self, skip: int, duration:float, dx:float=0.1, dy:float=0.1, sampling_frequency:float=48000, d0:int=6, l_prime:int=20 ) -> None:
        self.skip = skip
        self.simulator = FDTD2D(duration=duration,dx=dx,dy=dy,sampling_frequency=sampling_frequency)
        self.i = 0
        self.dx = dx
        self.dy = dy
        self.d0 = d0
        self.l_prime = l_prime


    def __getitem__(self, index: int) -> dict:
        r"""Get a data.

        t: time_step
        h: height
        w: weight

        Returns:
            new_data: dict
        """
        
        data = self.simulator()

        bnd = data["bnd"]  # (h, w)
        u = data["u"][0 :: self.skip]  # (t, h, w)
        t = data["t"][0 :: self.skip]  # (t,)
        u0 = u[0]

        bnd_pointscloud = np.argwhere(bnd == 1)  # (L_bnd, 2)
        bnd_set = set(map(tuple, bnd_pointscloud))  
        while True:
            i = random.randint(0, u.shape[1] - 1)  # i ∈ [0, height - 1]
            j = random.randint(0, u.shape[2] - 1)  # j ∈ [0, width - 1]
            if (i, j) not in bnd_set:
                break
            
        # x = self.dx * i
        # y = self.dy * j
        x = np.array(i).astype(np.float32) #()
        y = np.array(j).astype(np.float32)

        uxy = u[:,i,j] #(l,)

        data = {
                "dataset_name": "FDTD_2D_RIR_PointsCloud",
                "bnd": bnd[...], # (h, w)
                "u0": u0[None, ...], # (1, h, w)
                "x": x[None, ...], # (1,)
                "y": y[None, ...], # (1,)
                "uxy": uxy[..., None, None], # (l, 1, 1)
                "d0": self.d0,
                "l_prime": self.l_prime
        }

        return data

    def __len__(self) -> int:
        return 100 # Number of steps in an epoch, 1getitem = 1len

class FDTD2D:
    def __init__(
        self, 
        duration: float = 0.02, # total time
        dx:float=0.1,
        dy:float=0.1,
        sampling_frequency:int=None,
        verbose: bool = False
    ):
        r"""FDTD wave simulator."""

        self.duration = duration
        self.verbose = verbose
        
        self.dx = dx  # Should be smaller than λ
        self.dy = dy
        self.nx = 64
        self.ny = 64
        
        self.c = 343.
        if sampling_frequency == None:
            self.dt = self.dx / self.c / 3  # CFL condition
        else:
            self.dt = 1.0 / sampling_frequency
        self.nt = round(self.duration / self.dt)

    def __call__(self) -> dict:

        # Initialize
        u = np.zeros((self.nx, self.ny))
        u_prev = np.zeros_like(u)
        u_next = np.zeros_like(u)
        
        # Boundary and obstacles
        bnd = self.sample_boundary()  # shape: (x, y)
        
        # Sample source
        u = self.sample_source()  # shape: (x, y)
        
        us = []

        # Simulate
        for t in range(self.nt):

            us.append(u.copy())
            
            # Calculate u_next by discretized 2D wave equation
            laplacian = (1. / self.dx**2) * (
                u[2:, 1:-1] + u[:-2, 1:-1] + 
                u[1:-1, 2:] + u[1:-1, :-2] - 
                4 * u[1:-1, 1:-1]
            )

            u_next[1:-1, 1:-1] = 2 * u[1:-1, 1:-1] - u_prev[1:-1, 1:-1] + (self.c**2 * self.dt**2) * laplacian
            
            # Apply rigid boundary
            u_next[bnd] = 0
            u[bnd] = 0
            u_prev[bnd] = 0

            # Update state
            u_prev[:] = u
            u[:] = u_next
            
            if self.verbose:
                print("{}/{}".format(t, self.nt))

        us = np.array(us)
        ts = np.arange(self.nt) * self.dt
        
        data = {
            "bnd": bnd.astype(np.float32),
            "t": ts.astype(np.float32),
            "u": us.astype(np.float32),
        }

        return data

    def sample_boundary(self) -> np.ndarray:
        r"""Sample boundary"""

        bnd = np.zeros((self.nx, self.ny), dtype=bool)

        # Boundary
        bnd[0 : 2, :] = 1
        bnd[-2:, :] = 1
        bnd[:, 0 : 2] = 1
        bnd[:, -2:] = 1

        # Obstacles
        cx = round(0.25 * self.nx)  # center of obstacle
        cy = round(0.5 * self.ny)
        wx = random.randint(round(0.1 * self.nx), round(0.4 * self.nx))  # width of obstacle
        wy = random.randint(round(0.2 * self.ny), round(0.8 * self.ny))
        
        bnd[cx - wx // 2 : cx + wx // 2, cy - wy // 2 : cy + wy // 2] = 1

        return bnd

    def sample_source(self) -> np.ndarray:
        r"""Sample source."""

        cx = random.randint(32, 64)
        cy = random.randint(10, 54)
        x = np.arange(self.nx)[:, None]
        y = np.arange(self.ny)[None, :]
        sigma = .5
        u = np.exp(-((x - cx)**2 + (y - cy)**2) / (2*sigma**2))

        return u


def visualize(x: np.ndarray, bound: np.ndarray) -> None:
    r"""Visualize soundfild."""
    
    fig, ax = plt.subplots()
    cax = ax.imshow(x[0], cmap='jet', vmin=-1, vmax=1)
    fig.colorbar(cax)

    def update(frame):
        cax.set_data(frame)
        return cax,

    # from IPython import embed; embed(using=False); os._exit(0)
    frames = np.clip(x + bound, -1, 1)

    ani = animation.FuncAnimation(fig, func=update, frames=frames, interval=20)
    writer = animation.FFMpegWriter(fps=24, bitrate=5000)

    out_path = "out.mp4"
    ani.save(out_path, writer=writer)
    print("Write sound filed MP4 to {}".format(out_path))


if __name__ == "__main__":

    import matplotlib.animation as animation
    import matplotlib.pyplot as plt

    # FDTD simulator
    simulator = FDTD2D(duration=0.01, verbose=True)

    # Simulate
    t = time.time()
    data = simulator()
    simulation_time = time.time() - t

    # Print
    skip = 5
    u = data["u"][0 :: skip]  # (t, x, y)
    bound = data["bnd"]  # (x, y)
    print("Sound field shape: {}".format(u.shape))
    print("Simulation time: {:.2f} s".format(simulation_time))

    # Write video for visualization
    visualize(u, bound)