# Notas sobre HQ-SAM

## Treinamento

### Dataset

O dataset do SAM-HQ é composto por imagens de diferentes dimensões que são redimensionados para 1024x1024 usando interpolação bilinear. A flag `--input_size` permite mudar a resolução ao rodar o script de treino.

> Uma nota é que o dataset do sam original tem um formato diferente, a segmentação está em formato COCO RLE. Mais sobre [isso aqui](https://github.com/facebookresearch/segment-anything/tree/main#dataset).

Além de redimensionamento, há outros pré-processamentos: inversão aleatória de eixo horizontal e `LargeScaleJitter`.

O `LargeScaleJitter` faz basicamente 3 coisas:
1. Redimensiona aleatoriamente;
2. Corta aleatoriamente;
3. Preenche com cinza (padding);


```python
## train.py ln 323 ##

if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train")
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        training = True)
        print(len(train_dataloaders), " train dataloaders created")
```


```python
## os padroes da classe declarados antes do trecho selecinado abaixo ##
self = {"desired_size": 1024, "aug_scale_min": 0.1, "aug_scale_max": 2.0}

##  dataloader.py ln 164 ##
imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

#resize keep ratio

# essa variável abaixo é declarada, mas não é usada
# out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
scaled_size = (random_scale * self.desired_size).round()

scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
scaled_size = (image_size * scale).round().long()
        
scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        
# random crop
crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

margin_h = max(scaled_size[0] - crop_size[0], 0).item()
margin_w = max(scaled_size[1] - crop_size[1], 0).item()
offset_h = np.random.randint(0, margin_h + 1)
offset_w = np.random.randint(0, margin_w + 1)
crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

# pad
padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

```

As anotações do dataset são as máscaras de segmentação dos objetos. 
Não tem anotado especificamente os tokens de entrada (bounding box, pontos ou noise_mask).

O que acontece é que a partir das máscaras ground truth, são geradas em tempo de treino tokens de entrada aleatórios.


```python
## train.py ln 407 - ln 4016 ##
input_keys = ['box','point','noise_mask']
labels_box = misc.masks_to_boxes(labels[:,0,:,:])
try:
    labels_points = misc.masks_sample_points(labels[:,0,:,:])
except:
    # less than 10 points
    input_keys = ['box','noise_mask']
    labels_256 = F.interpolate(labels, size=(256, 256), mode='bilinear')
    labels_noisemask = misc.masks_noise(labels_256)
```

Após gerar os tokens, um tipo aleatório é escolhido para cada imagem:


```python
## train.py ln 417 - ln 434 ##
batched_input = []
for b_i in range(len(imgs)):
    dict_input = dict()
    input_image = torch.as_tensor(imgs[b_i].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous()
    dict_input['image'] = input_image 
    input_type = random.choice(input_keys)
    if input_type == 'box':
        dict_input['boxes'] = labels_box[b_i:b_i+1]
    elif input_type == 'point':
        point_coords = labels_points[b_i:b_i+1]
        dict_input['point_coords'] = point_coords
        dict_input['point_labels'] = torch.ones(point_coords.shape[1], device=point_coords.device)[None,:]
    elif input_type == 'noise_mask':
        dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
    else:
        raise NotImplementedError
    dict_input['original_size'] = imgs[b_i].shape[:2]
    batched_input.append(dict_input)
```

Pelo menos durante o treino, a capacidade do SAM de gerar múltiplas mascáras de saída é desligada.


```python
## train.py ln 436 - ln 437 ##
with torch.no_grad():
    batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
```
