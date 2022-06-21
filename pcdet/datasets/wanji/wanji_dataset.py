import copy
import pickle

import numpy as np
import pandas as pd
from skimage import io
from os import path
import open3d as o3d
# from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
# from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti, partial_dataset_utils
from ..dataset import DatasetTemplate


class WanjiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.pcd'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # self.pcd_path = path.join(self.root_path, ('training' if self.split != 'test' else 'testing'))
        self.pcd_path = path.join(self.root_path, 'pcd_rename')
        # print(self.pcd_path, self.root_path)

        split_dir = path.join(self.root_path, 'split', (self.split + '.txt'))
        print(split_dir)
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if path.exists(split_dir) else None

        self.ext = ext
        self.type2class={'bus':0, 'car':1, 'bicycle':2, 'pedestrian':3, 'tricycle':4, 'semitrailer':5, 'truck':6}
        self.class2type = {self.type2class[t]:t for t in self.type2class}

    def __len__(self):
        return len(self.sample_id_list)

    def __getitem__(self, index):
        # print(index)
        lidar_file = path.join(self.pcd_path, (self.sample_id_list[index]+self.ext))
        if self.ext == '.bin':
            points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(lidar_file)
        elif self.ext == '.pcd':
            points = self.load_pcd_data(lidar_file)
        else:
            raise NotImplementedError
        input_dict = {
            'points': points,
            'frame_id': self.sample_id_list[index],
        }
        # gt_boxes

        obj_dict = self.get_label(index)
        
        loc = obj_dict['loc'].values
        l, w, h = obj_dict['l'].values, obj_dict['w'].values, obj_dict['h'].values
        rots = np.transpose(obj_dict['ry'].values)[0]
        
        class_id = np.transpose(obj_dict['class_id'].values)[0]
        # print(class_id)
        gt_names = [self.class2type[id] for id in class_id]
        gt_boxes_lidar = np.concatenate([loc, l, w, h, (rots[..., np.newaxis])/180*np.pi], axis=1)
        input_dict.update({
            'gt_names': np.array(gt_names),
            'gt_boxes': gt_boxes_lidar
        })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict
    
    def get_label(self, idx):
        label_file = path.join(self.root_path, 'csv_rename' , ('%0004d.csv' % idx))
        # print(label_file.split('/')[-1])
        # label_file = path.join(self.root_path, 'csv' , ('%s.csv' % idx))
        assert path.exists(label_file)
        try:
            df = pd.read_csv(label_file, header=None)
        except:
            print(label_file)
            assert(0)
        target_id = df.iloc[:, [0]]
        class_id = df.iloc[:, [1]]
        loc = df.iloc[:, [2,3,4]]/100
        ry = df.iloc[:, [6]]
        l = df.iloc[:, [7]]/100
        w = df.iloc[:, [8]]/100
        h = df.iloc[:, [9]]/100
        
        return {
            'target_id': target_id,
            'class_id': class_id,
            'loc': loc,
            'ry': ry,
            'l': l,
            'w': w,
            'h': h
        }
        
    def load_pcd_data(self, file_path):
        pcd = o3d.t.io.read_point_cloud(file_path)
        position = pcd.point["positions"].numpy()
        intensity = pcd.point["intensity"].numpy()
        points = np.concatenate([position, intensity], axis=1, dtype=float)
        return points
    
    def old_load_pcd_data(self, file_path):
        # print(file_path.split('/')[-1])
        pts = []
        f = open(file_path, 'r')
        data = f.readlines()
        f.close()
        line = data[9]
        line = line.strip('\n')
        l = line.split(' ')
        pts_num = eval(l[-1])
        for line in data[11:]:
            line = line.strip('\n')
            xyzargb = line.split(' ')
            x, y, z, i = [eval(i) for i in xyzargb[:4]]
            # print(x,y,z,i)
            pts.append([x, y, z, i])

        assert len(pts) == pts_num
        res = np.zeros((pts_num, len(pts[0])), dtype=float)
        for j in range(pts_num):
            res[j] = pts[j]
        return res
    
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            # calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index]
            # pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_camera = pred_boxes
            # pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            #     pred_boxes_camera, calib, image_shape=image_shape
            # )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos
    
    def load_test_set_info(self, test_list):
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict
        res = []
        for idx in test_list:
            label_dict = self.get_label(int(idx))
        
            loc = label_dict['loc'].values
            l, w, h = label_dict['l'].values, label_dict['w'].values, label_dict['h'].values
            rots = np.transpose(label_dict['ry'].values)[0]
            
            class_id = np.transpose(label_dict['class_id'].values)[0]
            gt_names = [self.class2type[id] for id in class_id]
            gt_names = np.array(gt_names)
            gt_boxes_lidar = np.concatenate([loc, l, w, h, (rots[..., np.newaxis])/180*np.pi], axis=1)
            
            num_samples = gt_names.shape[0]
            item_dict = get_template_prediction(int(num_samples))
            if num_samples == 0:
                res.append(item_dict)
                continue
            item_dict['name'] = gt_names
            item_dict['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + gt_boxes_lidar[:, 6]
            item_dict['bbox'] = gt_boxes_lidar
            item_dict['dimensions'] = gt_boxes_lidar[:, 3:6]
            item_dict['location'] = gt_boxes_lidar[:, 0:3]
            item_dict['rotation_y'] = gt_boxes_lidar[:, 6]
            item_dict['boxes_lidar'] = gt_boxes_lidar
            res.append(item_dict)
        return res
    
    def evaluation(self, det_annos, class_names, eval_metric=None, eval_levels_list_cfg=None, **kwargs):
        # if 'annos' not in self.kitti_infos[0].keys():
        #     return None, {}

        from . import wanji_eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        test_list = [i['frame_id'] for i in eval_det_annos]
        
        # eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        eval_gt_annos = self.load_test_set_info(test_list)

        if eval_metric == 'wanji':
            ap_result_str, ap_dict = kitti_eval.get_wanji_eval_result(eval_gt_annos, eval_det_annos, class_names, eval_levels_list_cfg=eval_levels_list_cfg)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
    
    
        