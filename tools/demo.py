import argparse
import glob
from pathlib import Path
from os import path
import pandas as pd
# import mayavi.mlab as mlab
import numpy as np
import torch
from os import path
from threading import Thread
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
# from visual_utils import visualize_utils as V
import open3d
from visual_utils import open3d_vis_utils as V
OPEN3D_FLAG = True

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.pcd'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list.sort()
        self.sample_file_list = data_file_list
        self.label_list = []
        for item in self.sample_file_list:
            self.label_list.append(self.get_label(item))

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            points = self.load_pcd_data(self.sample_file_list[index])
        else:
            raise NotImplementedError
        input_dict = {
            'points': points,
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    def load_pcd_data(self, file_path):
        # print(file_path)
        pcd = open3d.t.io.read_point_cloud(path.join(file_path))
        position = pcd.point["positions"].numpy()
        intensity = pcd.point["intensity"].numpy()
        points = np.concatenate([position, intensity], axis=1, dtype=float)
        return points
    
    def get_label(self, label_file):
        pathlist = str(label_file).split("/")
        file_name = pathlist[-1][:-3]
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
        return gt_boxes_lidar, class_id, file_name
        

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.pcd', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def draw():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    boxes, class_id = demo_dataset.get_label(args.data_path)
    print(boxes.shape[0])
    ref_scores = [0.98 for i in range(boxes.shape[0])]
    ref_labels = [i+1 for i in class_id]
    print((ref_labels))
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=boxes,
                ref_scores=ref_scores, ref_labels=ref_labels, is_gt=True
            )
    

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of PDV-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')
    
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            print(args.data_path)
            ref_boxes_np=pred_dicts[0]['pred_boxes'].cpu().numpy()
            ref_scores_np=pred_dicts[0]['pred_scores'].cpu().numpy()
            ref_labels_np=pred_dicts[0]['pred_labels'].cpu().numpy()
            gt_boxes, gt_labels, file_name = demo_dataset.label_list[idx]

            print('gt: {} {}'.format(len(gt_labels), np.asarray(gt_labels, dtype=int)))
            print(print('pred: {} {}'.format(len(ref_labels_np), ref_labels_np)))
            
            V.draw_scenes(
                points=data_dict['points'][:, 1:], gt_boxes=gt_boxes, ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'], is_gt=file_name
            )
            
            # if not OPEN3D_FLAG:
            #     mlab.show(stop=True)

    logger.info('Demo done.')



if __name__ == '__main__':
    main()
    # draw()

