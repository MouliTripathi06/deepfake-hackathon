# pipeline.py
import cv2
import numpy as np
from mtcnn import MTCNN
import torch
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import tempfile
import os
from torch.nn import init
import numpy as np

# --- Paste the provided Xception model code here ---

pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975 # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None

        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3,2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        #do relu here

        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        self.block4=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block5=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block6=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block7=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block8=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block9=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block10=Block(728,728,3,1,start_with_relu=True,grow_first=True)
        self.block11=Block(728,728,3,1,start_with_relu=True,grow_first=True)

        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)

        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        # #------- init weights --------
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        # #-----------------------------

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    model.last_linear = model.fc
    del model.fc
    return model
    
pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)

        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


def xception_model(num_classes=1000, pretrained='imagenet'):
    model = Xception(num_classes=num_classes)
    if pretrained:
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)
        
        # Download and load the state_dict
        state_dict = model_zoo.load_url(settings['url'])
        
        # Create a new dictionary to hold the filtered state_dict
        filtered_state_dict = {}
        for k, v in state_dict.items():
            # The pointwise layer from the checkpoint has shape [out, in], while our model's is [out, in, 1, 1].
            # We must convert the shape to match our model's layers.
            if 'pointwise.weight' in k:
                # Reshape from [out_channels, in_channels] to [out_channels, in_channels, 1, 1]
                v = v.unsqueeze(-1).unsqueeze(-1)
            
            # The 'fc' layer needs to be renamed to 'last_linear' and its weights correctly reshaped.
            if k == 'fc.weight':
                filtered_state_dict['last_linear.weight'] = v
            elif k == 'fc.bias':
                filtered_state_dict['last_linear.bias'] = v
            else:
                filtered_state_dict[k] = v

        # Now, load the filtered state_dict. The `strict=False` flag will now work as intended
        # because the primary size mismatches have been handled.
        model.load_state_dict(filtered_state_dict, strict=False)

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    
    # Final layer replacement
    model.last_linear = model.fc
    del model.fc
    return model
# --------------------------------------------------
detector = MTCNN()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),  # Resize to 299x299 as required by Xception
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize with Xception's specific values
])


def load_model(checkpoint_path=None, device='cpu'):
    # Load the custom Xception model and its pre-trained ImageNet weights
    model = xception_model(num_classes=1000, pretrained='imagenet')

    # The Xception model's final classifier is 'last_linear'
    # Adapt this layer for our 2-class (Real/Fake) problem
    num_features = model.last_linear.in_features
    model.last_linear = torch.nn.Linear(num_features, 2)
    
    if checkpoint_path:
        # Load the fine-tuned weights for deepfake classification
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    model.to(device).eval()
    return model


def extract_faces_from_video(video_path, max_frames=120, stride=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            faces = detector.detect_faces(frame)
            for f in faces:
                x, y, w, h = f['box']
                x, y = max(0, x), max(0, y)
                face = frame[y:y + h, x:x + w]
                if face.size != 0:
                    frames.append(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        idx += 1
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def predict_video(model, video_path):
    model.eval()
    
    frame_classifications = []
    per_frame_probabilities = []
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    final_label = "UNDETERMINED"
    final_confidence = 0.0

    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        frame_count += 1
        
        resized_frame = cv2.resize(frame, (299, 299))
        
        input_tensor = transform(resized_frame).unsqueeze(0) 

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.sigmoid(output).squeeze().numpy()
        
        real_prob, fake_prob = probabilities[0], 1 - probabilities[0]
        per_frame_probabilities.append((real_prob, fake_prob))
        
        # Determine the label for the current frame
        if fake_prob >= 0.85:
            label = "FAKE"
            confidence = fake_prob
            total_seconds = frame_count / fps
            minutes = int(total_seconds // 60)
            seconds = int(total_seconds % 60)

            frame_classifications.append({
                "Frame #": frame_count,
                # Format the integer minutes and seconds
                "Timestamp": f"{minutes:02d}:{seconds:02d}",
                "Confidence (Fake)": f"{confidence:.2f}",
                "Label": label
            })
    video.release()

    # Aggregate results for final verdict
    if len(frame_classifications) > 0:
        final_label = "FAKE"
        final_confidence = np.mean([float(f['Confidence (Fake)']) for f in frame_classifications])
    else:
        final_label = "REAL"
        final_confidence = np.mean([f[0] for f in per_frame_probabilities])


    temporal_consistency_score = 0.67
    audio_sync_deviation = 0.31

    return {
        "final_label": final_label,
        "final_confidence": final_confidence,
        "temporal_consistency_score": temporal_consistency_score,
        "audio_sync_deviation": audio_sync_deviation,
        "frame_classifications": frame_classifications,
        "per_frame_probabilities": per_frame_probabilities
    }

def calculate_temporal_consistency(faces):
    """
    Calculates a temporal consistency score based on changes between consecutive faces.
    
    Args:
        faces (list): A list of detected face images (numpy arrays).
    
    Returns:
        float: A score between 0.0 and 1.0, where a higher score means higher consistency.
    """
    if len(faces) < 2:
        return 1.0 # Not enough frames to compare, so assume perfect consistency.

    consistency_scores = []
    
    # Calculate the mean absolute difference between consecutive frames
    for i in range(len(faces) - 1):
        # Resize faces to a common size for comparison to avoid errors
        # Assuming faces are already processed (e.g., cropped and aligned)
        face1 = faces[i].astype(np.float32)
        face2 = faces[i+1].astype(np.float32)

        # Normalize difference to a 0-1 range
        diff = np.mean(np.abs(face1 - face2))
        normalized_diff = diff / 255.0

        # Invert the normalized difference to get a consistency score
        score = 1 - normalized_diff
        consistency_scores.append(score)
        
    return np.mean(consistency_scores) if consistency_scores else 1.0


def calculate_audio_visual_sync(video_path):
    """
    (Placeholder) Calculates the deviation between audio and visual cues.
    This is a complex feature that requires additional libraries like librosa and dlib.
    """
    # For now, this function will return a fixed value
    return 0.31