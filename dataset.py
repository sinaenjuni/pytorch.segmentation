import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import os

class HICO(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_path, select_obj, transform, mode):
        self.image_dir = image_dir
        self.annotation_path = annotation_path
        self.select_obj = select_obj
        self.transform = transform
        self.train_dataset = []
        self.test_dataset = []

        self.num_images = 0
        self.mode = mode
        self.onehot = np.eye(5)
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        annotations = pd.read_csv(self.annotation_path, index_col=0)
        annotations = annotations.values.tolist()

        for line in annotations:
            if line[8] not in self.select_obj:
                continue

            img_id = line[0]
            label = line[-1]
            types = line[9]
            self.train_dataset.append([types, img_id, label])

        self.test_dataset = self.train_dataset[:2000]
        self.train_dataset = self.train_dataset[2000:]

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        types, filename, label = dataset[index]
        label = self.onehot[int(label)]
        filename = str(filename).zfill(8)
        filename = f'{types}2015/HICO_{types}2015_{filename}.jpg'
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.FloatTensor(label)


def getLoader(image_dir, annotation_path, select_obj,
              crop_size=512, image_size=128,
              batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    transform = []
    transform.append(T.Resize((image_size, image_size)))
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    #     transform.append(T.CenterCrop(crop_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = HICO(image_dir=image_dir,
                   annotation_path=annotation_path,
                   select_obj=select_obj,
                   transform=transform,
                   mode=mode)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=(mode == 'train'),
                                              num_workers=num_workers)
    return data_loader



if __name__ == '__main__':
    data_loader = getLoader(image_dir='data/Data_hico/', select_obj=['boat'],
                            annotation_path='data/Data_hico/hico_annotations.csv')
    print(len(data_loader))
