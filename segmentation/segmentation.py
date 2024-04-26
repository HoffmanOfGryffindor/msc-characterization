import numpy as np
import platform, os
from glob import glob
import scipy, cv2
import mahotas
import yaml
from sklearn.model_selection import train_test_split
from readlif.reader import LifFile
from math import floor, ceil

class ImageSegmenter:
    def __init__(self, blur_sigma=0.5):
        self.sigma = blur_sigma

        data_config = os.path.join(os.path.dirname(__file__), "../configs/data.yaml")
        with open(data_config, "rb") as file:
            config = yaml.safe_load(file)
        
        self.config = config

    def __call__(self, directory: str, save: bool = True):
        print("-"*5 + "Reading LIF Files" + "-"*5)
        images = self.read_lifs(directory=directory)
        print("-"*5 + "Extracting Nuceli" + "-"*5)
        data = self.get_nucleus(images)
        print("-"*5 + "Saving Nuclei to Files" + "-"*5)
        self.save_imagery(data)

    # converts 16 bit images into 8 bit grayscale images
    def convert_grayscale(self, img):
        img = img.astype(np.float32)
        img = 255 * (img - np.amin(img))/(np.amax(img)-np.amin(img))
        img = img.astype(np.uint8)

        return img if img.ndim == 2 else img[0, ...]
    
    def convert_label(self, label: str) -> str:
        return self.config['labels'][label]

    def read_lifs(self, directory: str, folder: bool = False) -> dict:
        cells = {}
        search = os.path.join(directory, "*.lif") if folder else os.path.join(directory, "**/*.lif")
        for file in glob(search, recursive=True):
            file_name = file
            *_, directory, keys = file_name.split()[0].split('\\') if platform.system() == "Windows" else file_name.split('/')
            _, time, *_ = keys.split('_')[:-1]
            channels = self.config['exceptions'][keys[:-4]] if keys[:-4] in self.config['exceptions'].keys() else self.config[directory] 
            meta = {
                'time': self.convert_label(time),
                'channels': channels,
                'file': keys,
            }
            cells = self.extract_files(file_name, cells, meta)
        
        for chan in cells.keys():
            cells[chan]['img'] = np.array([self.convert_grayscale(np.array(img)) for img in cells[chan]['img']])
    
        return cells
    
    def extract_files(self, file: str, cells: dict, meta: dict) -> dict:
        lif = LifFile(file)
        for x in lif.get_iter_image():
            for idx, chan in enumerate(meta["channels"]):
                try:
                    if chan in cells.keys():
                        cells[chan]['img'].extend([np.array(frame) for frame in x.get_iter_t(c=idx)])
                        cells[chan]['time'].extend([meta["time"] for _ in x.get_iter_t(c=idx)])
                        cells[chan]['file'].extend([meta["file"] for _ in x.get_iter_t(c=idx)])
                    else:
                        cells.update(
                            {
                            chan: {
                                'img': [np.array(frame) for frame in x.get_iter_t(c=idx)],
                                'time': [meta["time"] for _ in x.get_iter_t(c=idx)],
                                'file': [meta["file"] for _ in x.get_iter_t(c=idx)]
                                }
                        }
                    ) 
                except:
                    continue
        return cells
    
    def find_contours(self, image) -> np.ndarray:

        image = scipy.ndimage.gaussian_filter(image, sigma=self.sigma)
        _, thresh = cv2.threshold(src=image, dst=np.copy(image), thresh=50, maxval=255, type=cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        
        return contours
    
    def area(self, cnt: np.array, img_shape: tuple) -> int:
        img = np.zeros(img_shape)
        cnt[:, :, 0] -= cnt[:, :, 0].min()
        cnt[:, :, 1] -= cnt[:, :, 1].min()
        cv2.fillPoly(img, pts=[cnt], color=1)
        return np.sum(img.flatten())


    def crop(self, image: np.ndarray) -> np.ndarray:
        contours = self.find_contours(image)
        y_len, x_len = image.shape
        bbox, areas = None, np.array([])
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = self.area(contour, (h, w))
            if x in [0, x_len-w] or y in [0, y_len-h] or area > 4000 or area < 250: continue
            areas = np.append(areas, area)
            if bbox is None:
                bbox = np.array([x, y, w, h])
            else:
                bbox = np.vstack((bbox, np.array([x, y, w, h])))
    
        return bbox, areas
    
    def get_nucleus(self, images: dict, stain_name : str = "hoechst") -> list[dict]:
        
        nuclei = []

        for idx, img in enumerate(images[stain_name]['img']):
            bboxes, areas = self.crop(img)
            if bboxes is None or areas is None or bboxes.ndim == 1: continue
            for bbox, area in zip(bboxes, areas):
                x, y, w, h = bbox
                nucleus = {
                    "nucleus": img[y:y+h, x:x+w],
                    "actin": images["actin"]['img'][idx][y:y+h, x:x+w],
                    "brightfield": images["brightfield"]['img'][idx][y:y+h, x:x+w],
                    "bbox": bbox,
                    "time": images[stain_name]['time'][idx],
                    "file": images[stain_name]['file'][idx],
                    "area": area,
                    "haralick": mahotas.features.haralick(img[y:y+h, x:x+w]).mean(0),
                }

                nuclei.append(nucleus)
        
        return nuclei
    
    def augment_imagery(self, data: dict):
        max_dim = self.find_largest_dim(data)
        images = np.array([self.condense_imagery(x['nucleus'], x['actin'], x['brightfield'], max_dim) for x in data])
        labels = np.array([x['time'] for x in data])

        x_train, x_test, y_train, y_test = train_test_split(images, labels)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
        sets = map(self.rotate_images, [x_train, x_val, x_test], [y_train, y_val, y_test])
        
        return list(sets)
    
    def rotate_images(self, images: np.ndarray, labels: np.ndarray):
        for img, label in zip(images, labels):
            for _ in range(3):
                img = np.rot90(img)
                images = np.append(images, img[None, ...], axis=0)
                labels = np.append(labels, label)

        return (images, labels)
    
    def pad_image(self, image: np.ndarray, max_dim: int):
        img_h, img_w = image.shape
        w_pad = [int((max_dim-img_w)/2)] * 2 if (max_dim-img_w)/2 % 2 == 0 else [floor((max_dim-img_w)/2), ceil((max_dim-img_w)/2)]
        h_pad = [int((max_dim-img_h)/2)] * 2 if (max_dim-img_h)/2 % 2 == 0 else [floor((max_dim-img_h)/2), ceil((max_dim-img_h)/2)]
        return np.pad(image, (h_pad, w_pad), "constant")

    def condense_imagery(self, nucleus: np.ndarray, actin: np.ndarray, bright: np.ndarray, max_dim: int):
        nucleus, actin, bright = map(self.pad_image, [nucleus, actin, bright], [max_dim]*3)
        return np.dstack((nucleus, actin, bright))

    def find_largest_dim(self, nuclei: dict):
        max_dim = 0
        for x in nuclei:
            *_, w, h = x['bbox']
            if max_dim < w: max_dim = w
            if max_dim < h: max_dim = h

        return max_dim

    def save_imagery(self, nuclei: dict) -> None:
        sets = self.augment_imagery(nuclei) # sets contains train, val, test images and labels
        for (s, name) in zip(sets, ["train", "val", "test"]):
            for data, data_type in zip(s, ['x','y']):
                with open(f'{data_type}_{name}.npy', 'wb') as f:
                    np.save(f, data)
                f.close()