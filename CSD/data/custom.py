
import sys

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).parent.resolve()))

class CustomD(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.pathlist = [x for x in self.root_dir.iterdir() if Path.is_file(x) and x.stem != 'clipscores' and x.stem != 'rewards']
        self.namelist = [x.stem for x in self.pathlist]

        # assert osp.exists(osp.join(root_dir, 'wikiart.csv'))
        # annotations = vx.from_csv(f'{self.root_dir}/wikiart.csv')
        # acceptable_artists = list(set(annotations[annotations['split'] == 'database']['artist'].tolist()))
        # temprepo = annotations[annotations['artist'].isin(acceptable_artists)]
        # self.pathlist = temprepo[temprepo['split'] == split]['path'].tolist()

        # self.namelist = list(map(lambda x: x.split('/')[-1], self.pathlist))

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        img_loc = self.pathlist[idx]  # os.path.join(self.root_dir, self.split,self.artists[idx] ,self.pathlist[idx])
        image = Image.open(img_loc).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, idx