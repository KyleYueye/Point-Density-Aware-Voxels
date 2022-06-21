import open3d as o3d
from visual_utils import open3d_vis_utils as V
import numpy as np
import pandas as pd
import struct
from os import path

class Display():
    def __init__(self, file, ext='.pcd'):
        self.ext = ext
        self.file = path.join(file)
        self.read_pc()
        # print(self.points)
    
    def load_pcd_data(self, file_path):
        pcd = o3d.t.io.read_point_cloud(file_path)
        position = pcd.point["positions"].numpy()
        intensity = pcd.point["intensity"].numpy()
        points = np.concatenate([position, intensity], axis=1, dtype=float)
        return points
    
    def read_bin_velodyne(self, file_path):
        pc_list=[]
        with open(file_path,'rb') as f:
            content=f.read()
            pc_iter=struct.iter_unpack('fffff',content)
            for idx,point in enumerate(pc_iter):
                pc_list.append([point[0],point[1],point[2]])
        return np.asarray(pc_list,dtype=np.float32)


    def read_pc(self):
        print("Reading point cloud...")
        if self.ext == '.bin':
            # self.points = np.fromfile(self.file, dtype=np.float32).reshape(-1, 4)
            self.points = self.read_bin_velodyne(self.file)
        elif self.ext == '.npy':
            self.points = np.load(self.file, allow_pickle=True)
            print(self.points.shape)
        elif self.ext == '.pcd':
            self.points = self.load_pcd_data(self.file)
        else:
            raise NotImplementedError
    def draw(self):
        V.draw_scenes(
            points=self.points, ref_boxes=None,
            ref_scores=None, ref_labels=None, is_gt=False
        )
    def draw_with_label(self):
        gt_boxes, gt_label = self.get_label(self.file)
        V.draw_scenes(
            points=self.points, ref_boxes=gt_boxes,
            ref_scores=None, ref_labels=np.asarray(gt_label+1,dtype=int), is_gt=True
        )
    def get_label(self, label_file):
        pathlist = str(label_file).split("/")
        pathlist[-2] = 'csv_rename'
        pathlist[-1] = pathlist[-1][:-3]+"csv"
        file = '/'.join(pathlist)
        print('/'.join(pathlist))
        assert path.exists(file)
        df = pd.read_csv(file, header=None)
        # print(df)
        target_id = df.iloc[:, [0]]
        class_id = np.transpose(df.iloc[:, [1]].values)[0]
        loc = df.iloc[:, [2,3,4]]/100
        ry = df.iloc[:, [6]]
        l = df.iloc[:, [7]]/100
        w = df.iloc[:, [8]]/100
        h = df.iloc[:, [9]]/100

        rots = np.transpose(ry.values)[0]
        gt_boxes_lidar = np.concatenate([loc, l, w, h, (rots[..., np.newaxis])/180*np.pi], axis=1)
        return gt_boxes_lidar, class_id


if __name__ == '__main__':
    # file = '/media/disk/02drive/nuScenes_dataset/nuScenes/sweeps/LIDAR_TOP/n015-2018-07-27-11-36-48+0800__LIDAR_TOP__1532662959398433.pcd.bin'
    file = '/media/disk/02drive/02xiangyu/MotionNet/nu_lidar_preprocessed/train/343_2/1.npy'
    d = Display(file, ext='.npy')
    d.draw()