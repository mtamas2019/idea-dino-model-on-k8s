import os
import torch
import numpy as np
from main import build_model_main
from util.slconfig import SLConfig
from PIL import Image
import datasets.transforms as T
import warnings
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn


app = FastAPI()

@app.post("/inference")
async def inference(image: UploadFile = File(...)):
    try:
        # Check GPU availability
        if not torch.cuda.is_available():
            return JSONResponse(content={'error': 'GPU not available'})

        # Load the model checkpoint
        model_config_path = os.getenv("MODEL_CONFIG_PATH", 'assets/DINO_4scale.py')
        model_checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH", "assets/checkpoint0023_4scale.pth")
        image = Image.open(image.file).convert("RGB")

        args = SLConfig.fromfile(model_config_path)
        args.device = torch.device('cuda')
        model, criterion, postprocessors = build_model_main(args)
        checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        _ = model.eval()

        # Transform the image
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image, _ = transform(image, None)

        # Perform inference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            output = model.cuda()(image[None].cuda())
            torch.cuda.empty_cache()

        output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]

        # Convert numpy arrays to lists
        output['scores'] = output['scores'].cpu().numpy().tolist()
        output['labels'] = output['labels'].cpu().numpy().tolist()
        output['boxes'] = output['boxes'].cpu().numpy().tolist()

        return JSONResponse(content=output)
    except Exception as e:
        return JSONResponse(content={'error': str(e)})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port)
