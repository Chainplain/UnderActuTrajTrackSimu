import scipy.io
import numpy as np

class traj_reader():
    def __init__(self, traj_name):
        self. name = traj_name
        poly_order = 6
        dim = 3

        mat_data = scipy.io.loadmat(traj_name)

        Coefs = mat_data['coef'].flatten().tolist()
        TimePerSeg = mat_data['T'].flatten().tolist()
        self. time_per_seg = TimePerSeg[0]
        

        Coefs_num = len(Coefs)
        self. segments_num = int(np.ceil(Coefs_num / (poly_order + 1) / dim))

        self. Total_time = self. time_per_seg * self. segments_num

        coef_mul = [i for i in range(poly_order, -1, -1)]
        coef_mul_high = coef_mul[1:]

        self.x_funs = []
        self.y_funs = []
        self.z_funs = []

        self.vx_funs = []
        self.vy_funs = []
        self.vz_funs = []

        self.ax_funs = []
        self.ay_funs = []
        self.az_funs = []

        poly_no = poly_order + 1


        for i in range(self. segments_num ):
            ## X:
            x_Coefs = Coefs[i * dim * poly_no + 0 : i * dim * poly_no + poly_no]
            vx_Coefs_raw = [x_Coefs * coef_mul for x_Coefs, coef_mul in zip(x_Coefs, coef_mul)]
            vx_Coefs = vx_Coefs_raw[:-1]            
            ax_Coefs_raw = [vx_Coefs * coef_mul_high for vx_Coefs, \
                            coef_mul_high in zip(vx_Coefs, coef_mul_high)]
            ax_Coefs = ax_Coefs_raw[:-1]

            x_traj = np.poly1d(x_Coefs)
            self.x_funs. append(x_traj)
            vx_traj = np.poly1d(vx_Coefs)
            self.vx_funs. append(vx_traj)
            ax_traj = np.poly1d(ax_Coefs)
            self.ax_funs. append(ax_traj)


            ## Y:
            y_Coefs = Coefs[i * dim * poly_no + poly_no : i * dim * poly_no + 2 * poly_no]
            vy_Coefs_raw = [y_Coefs * coef_mul for y_Coefs, coef_mul in zip(y_Coefs, coef_mul)]
            vy_Coefs = vy_Coefs_raw[:-1]            
            ay_Coefs_raw = [vy_Coefs * coef_mul_high for vy_Coefs, \
                            coef_mul_high in zip(vy_Coefs, coef_mul_high)]
            ay_Coefs = ay_Coefs_raw[:-1]

            y_traj = np.poly1d(y_Coefs)
            self.y_funs. append(y_traj)
            vy_traj = np.poly1d(vy_Coefs)
            self.vy_funs. append(vy_traj)
            ay_traj = np.poly1d(ay_Coefs)
            self.ay_funs. append(ay_traj)


            ## Z:
            z_Coefs = Coefs[i * dim * poly_no + 2 * poly_no : i * dim * poly_no + 3 * poly_no]
            vz_Coefs_raw = [z_Coefs * coef_mul for z_Coefs, coef_mul in zip(z_Coefs, coef_mul)]
            vz_Coefs = vz_Coefs_raw[:-1]
            az_Coefs_raw = [vz_Coefs * coef_mul_high for vz_Coefs, \
                            coef_mul_high in zip(vz_Coefs, coef_mul_high)]
            az_Coefs = az_Coefs_raw[:-1]

            z_traj = np.poly1d(z_Coefs)
            self.z_funs. append(z_traj)
            vz_traj = np.poly1d(vz_Coefs)
            self.vz_funs. append(vz_traj)
            az_traj = np.poly1d(az_Coefs)
            self.az_funs. append(az_traj)
    
    def get_x_pos(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.x_funs[seg_no](time - self. time_per_seg * seg_no)
    
    def get_y_pos(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.y_funs[seg_no](time - self. time_per_seg * seg_no)
    
    def get_z_pos(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.z_funs[seg_no](time - self. time_per_seg * seg_no)
    
    def get_x_vel(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.vx_funs[seg_no](time - self. time_per_seg * seg_no)
    
    def get_y_vel(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.vy_funs[seg_no](time - self. time_per_seg * seg_no)
    
    def get_z_vel(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.vz_funs[seg_no](time - self. time_per_seg * seg_no)
    
    def get_x_acc(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.ax_funs[seg_no](time - self. time_per_seg * seg_no)
    
    def get_y_acc(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.ay_funs[seg_no](time - self. time_per_seg * seg_no)
    
    def get_z_acc(self, time):
        if (time < 0) or (time >= self. time_per_seg  * self. segments_num):
            return None
        seg_no = int(np.floor(time / self. time_per_seg))
        return self.az_funs[seg_no](time - self. time_per_seg * seg_no)
        