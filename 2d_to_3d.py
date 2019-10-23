import torch
import torch.nn.init as init

class d_gen(nn.Module):
    def __init__(self):
        self.encoder=Encoder()
        self.decoder= Decoder()

    def forward(self,image):
        shape=self.encoder(image)
        dim,mask=self.decoder(shape)
        return dim,mask

model = d_gen()

optimizer = optim.SGD(model.parameters())
# 2D projections from predetermined viewpoints
dim, mask = model(RGB_images)
# fused point cloud
#fuseTrans is predetermined viewpoints info
dimid, ML = fuse3D(dim, mask, fuseTrans)
# Render new depth images at novel viewpoints
# renderTrans is novel viewpoints info
newDepth, newMask, collision = render2D(dimid, ML, renderTrans)
# Compute loss between novel view and ground truth
loss_depth = L1Loss()(newDepth, GTDepth)
loss_mask = BCEWithLogitLoss()(newMask, GTMask)
loss_total = loss_depth + loss_mask
# Back-propagation to update Structure Generator
loss_total.backward()
optimizer.step()
    