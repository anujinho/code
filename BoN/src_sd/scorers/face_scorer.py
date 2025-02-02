# ---------------------------------------------------------------------------
# Contains the code for Face Matching
# ---------------------------------------------------------------------------

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceRecognitionScorer(nn.Module):

    def __init__(self, fr_crop=False, mtcnn_face=False):
        super().__init__()
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        print(self.resnet)
        self.mtcnn = MTCNN(device='cuda')
        self.crop = fr_crop
        self.output_size = 160
        self.mtcnn_face = mtcnn_face

        self.target = None

    def extract_face(self, imgs, batch_boxes, mtcnn_face=False):
        image_size = imgs.shape[-1]
        faces = []
        for i in range(imgs.shape[0]):
            img = imgs[i]
            if not mtcnn_face:
                box = [48, 48, 208, 208]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            elif batch_boxes[i] is not None:
                box = batch_boxes[i][0]
                margin = [
                    self.mtcnn.margin * (box[2] - box[0]) / (self.output_size - self.mtcnn.margin),
                    self.mtcnn.margin * (box[3] - box[1]) / (self.output_size - self.mtcnn.margin),
                ]

                box = [
                    int(max(box[0] - margin[0] / 2, 0)),
                    int(max(box[1] - margin[1] / 2, 0)),
                    int(min(box[2] + margin[0] / 2, image_size)),
                    int(min(box[3] + margin[1] / 2, image_size)),
                ]
                crop_face = img[None, :, box[1]:box[3], box[0]:box[2]]
            else:
                # crop_face = img[None, :, :, :]
                return None

            faces.append(F.interpolate(crop_face, size=self.output_size, mode='bicubic'))
        new_faces = torch.cat(faces)

        return (new_faces - 127.5) / 128.0

    def get_faces(self, x, mtcnn_face=False):
        img = (x + 1.0) * 0.5 * 255.0
        img = img.permute(0, 2, 3, 1)
        with torch.no_grad():
            batch_boxes, batch_probs, batch_points = self.mtcnn.detect(img, landmarks=True)
            # Select faces
            batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
                batch_boxes, batch_probs, batch_points, img, method=self.mtcnn.selection_method
            )

        img = img.permute(0, 3, 1, 2)
        faces = self.extract_face(img, batch_boxes, mtcnn_face)
        return faces

    def forward(self, x, return_faces=False, mtcnn_face=None):
        x = TF.resize(x, (256, 256), interpolation=TF.InterpolationMode.BICUBIC)

        if mtcnn_face is None:
            mtcnn_face = self.mtcnn_face

       

        ######## Changes starts >>>>>>>>>>>>>>>>>>>>>>>>

        faces = None
        if not self.crop:
            out = self.resnet(x)
        else:
            faces = self.get_faces(x, mtcnn_face=mtcnn_face)
            if faces is None:
                return faces
            out = self.resnet(faces)

        if return_faces:
            return out, faces
        else:
            return out
        
        ####### Changes ends <<<<<<<<<<<<<<<<<<<<<<<<<<

    def cuda(self):
        self.resnet = self.resnet.cuda()
        self.mtcnn = self.mtcnn.cuda()
        return self
    
    def l1_loss(self, input, target):
        l = torch.abs(input - target).mean(dim=[1])
        return l
    
    def score(self, im_pix, target):

        if isinstance(im_pix, Image.Image):
            im_pix = transforms.ToTensor()(im_pix)
            im_pix = im_pix.unsqueeze(0)

        if self.target is None:
            with torch.no_grad():
                self.target = self(target.unsqueeze(0))

        # print(self.target.shape)
        # print(f'im_pix {im_pix.shape}')
        curr_target = self.target.repeat(im_pix.shape[0], 1)
        # print(f'curr_target {curr_target.shape}')
        curr_input = self(im_pix)
        # print(f'curr_input {curr_input.shape}')

        return - self.l1_loss(curr_input, curr_target)
    
    def reset_target(self):
        self.target = None
    
    def loss_fn(self, im_pix, target):

        return  - self.score(im_pix, target)
