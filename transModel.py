import torch
import torch.nn as nn
import torch.nn.functional as F

class gradientLoss(nn.Module):
    """
    Smoothness regularization loss on the flow field.
    """
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        # input: [B, 2, H, W] flow
        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if self.penalty == "l2":
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dH) + torch.mean(dW)) / 2.0
        return loss

class crossCorrelation3D(nn.Module):
    """
    Local (2D) cross correlation loss to measure similarity between moving and fixed images.
    """
    def __init__(self, in_ch, kernel=(9, 9), voxel_weights=None):
        super(crossCorrelation3D, self).__init__()
        self.in_ch = in_ch
        self.kernel = kernel
        self.voxel_weight = voxel_weights
        # For simplicity, we place the filter on GPU:0.
        # Change `.cuda(0)` to `.to(device)` in your actual code if needed.
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1]])).cuda(0)

    def forward(self, input, target):
        # input: deformed image
        # target: fixed image

        min_val, max_val = -1, 1
        target = (target - min_val) / (max_val - min_val)  # normalize target to [0,1]

        II = input * input
        TT = target * target
        IT = input * target

        pad = (int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2))
        T_sum = F.conv2d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv2d(input, self.filt, stride=1, padding=pad)
        TT_sum = F.conv2d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv2d(II, self.filt, stride=1, padding=pad)
        IT_sum = F.conv2d(IT, self.filt, stride=1, padding=pad)

        kernelSize = self.kernel[0] * self.kernel[1]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        cross = IT_sum - Ihat * T_sum - That * I_sum + That * Ihat * kernelSize
        T_var = TT_sum - 2 * That * T_sum + That * That * kernelSize
        I_var = II_sum - 2 * Ihat * I_sum + Ihat * Ihat * kernelSize
        cc = cross * cross / (T_var * I_var + 1e-5)

        loss = -1.0 * torch.mean(cc)
        return loss

class Dense3DSpatialTransformer(nn.Module):
    """
    Customized 2D spatial transformer (dense displacement).
    """
    def __init__(self):
        super(Dense3DSpatialTransformer, self).__init__()

    def forward(self, input1, input2):
        return self._transform(input1, input2)

    def _transform(self, input1, input2):
        batchSize = 1
        hgt = input2[:, 0].shape[1]
        wdt = input2[:, 1].shape[2]

        H_mesh, W_mesh = self._meshgrid(hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, hgt, wdt)
        H_upmesh = input2[:, 0] + H_mesh
        W_upmesh = input2[:, 1] + W_mesh
        
        return self._interpolate(input1, H_upmesh, W_upmesh)

    def _meshgrid(self, hgt, wdt):
        # note: you might need to pass device info if not always on cuda(0)
        h_t = torch.matmul(
            torch.linspace(0.0, hgt - 1.0, hgt).unsqueeze_(1),
            torch.ones((1, wdt))
        ).cuda(0)

        w_t = torch.matmul(
            torch.ones((hgt, 1)),
            torch.linspace(0.0, wdt - 1.0, wdt).unsqueeze_(1).transpose(1, 0)
        ).cuda(0)

        return h_t, w_t

    def _interpolate(self, input, H_upmesh, W_upmesh):
        nbatch = 1
        nch = 1
        height = 128
        width = 128

        # Expand input with 1-pixel padding on each side
        # so that indexing won't go out of bounds
        img = torch.zeros([nbatch, nch, height + 2, width + 2], dtype=torch.float).cuda(0)
        img[:, :, 1:-1, 1:-1] = input

        imgHgt = img.shape[2]
        imgWdt = img.shape[3]

        H_upmesh = H_upmesh.view(-1).float() + 1.0
        W_upmesh = W_upmesh.view(-1).float() + 1.0

        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        hf = torch.clamp(hf, 0, imgHgt - 1)
        hc = torch.clamp(hc, 0, imgHgt - 1)
        wf = torch.clamp(wf, 0, imgWdt - 1)
        wc = torch.clamp(wc, 0, imgWdt - 1)

        rep = torch.ones([height * width, ]).unsqueeze_(1).transpose(1, 0).cuda(0)
        bDHW = torch.matmul(
            (torch.arange(0, nbatch).float() * imgHgt * imgWdt).unsqueeze_(1).cuda(0),
            rep
        ).view(-1).int()

        W = imgWdt
        idx_000 = bDHW + hf * W + wf
        idx_010 = bDHW + hc * W + wf
        idx_001 = bDHW + hf * W + wc
        idx_011 = bDHW + hc * W + wc

        img_flat = img.view(-1, nch).float()
        val_000 = torch.index_select(img_flat, 0, idx_000.long())
        val_010 = torch.index_select(img_flat, 0, idx_010.long())
        val_001 = torch.index_select(img_flat, 0, idx_001.long())
        val_011 = torch.index_select(img_flat, 0, idx_011.long())

        dHeight = hc.float() - H_upmesh
        dWidth = wc.float() - W_upmesh

        wgt_000 = (dWidth * dHeight).unsqueeze_(1)
        wgt_010 = (dWidth * (1 - dHeight)).unsqueeze_(1)
        wgt_001 = ((1 - dWidth) * dHeight).unsqueeze_(1)
        wgt_011 = ((1 - dWidth) * (1 - dHeight)).unsqueeze_(1)

        output = (
            val_000 * wgt_000
            + val_010 * wgt_010
            + val_001 * wgt_001
            + val_011 * wgt_011
        )
        output = output.view(nbatch, height, width, nch).permute(0, 3, 1, 2)
        return output

class Cblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Cblock, self).__init__()
        self.block = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)

    def forward(self, x):
        return self.block(x)

class CRblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(CRblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.block(x)

class inblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super(inblock, self).__init__()
        self.block = CRblock(in_ch, out_ch, stride=stride)

    def forward(self, x):
        return self.block(x)

class outblock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2, output_padding=1):
        super(outblock, self).__init__()
        self.block = nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride)

    def forward(self, x):
        x = self.block(x)
        return x

class downblock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downblock, self).__init__()
        self.block = CRblock(in_ch, out_ch, stride=2)

    def forward(self, x):
        return self.block(x)

class upblock(nn.Module):
    def __init__(self, in_ch, CR_ch, out_ch):
        super(upblock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_ch, in_ch, 3, padding=1, stride=2, output_padding=1)
        self.block = CRblock(CR_ch, out_ch)

    def forward(self, x1, x2):
        upconved = self.upconv(x1)
        x = torch.cat([x2, upconved], dim=1)
        return self.block(x)

class registUnetBlock(nn.Module):
    """
    A registration UNet producing a flow field that warps x_start to x_T.
    """
    def __init__(self, input_nc, encoder_nc, decoder_nc):
        super(registUnetBlock, self).__init__()
        # encoder
        self.inconv = inblock(input_nc, encoder_nc[0], stride=1)
        self.downconv1 = downblock(encoder_nc[0], encoder_nc[1])
        self.downconv2 = downblock(encoder_nc[1], encoder_nc[2])
        self.downconv3 = downblock(encoder_nc[2], encoder_nc[3])
        self.downconv4 = downblock(encoder_nc[3], encoder_nc[4])

        # decoder
        self.upconv1 = upblock(encoder_nc[4], encoder_nc[4] + encoder_nc[3], decoder_nc[0])
        self.upconv2 = upblock(decoder_nc[0], decoder_nc[0] + encoder_nc[2], decoder_nc[1])
        self.upconv3 = upblock(decoder_nc[1], decoder_nc[1] + encoder_nc[1], decoder_nc[2])
        self.keepblock = CRblock(decoder_nc[2], decoder_nc[3])
        self.upconv4 = upblock(decoder_nc[3], decoder_nc[3] + encoder_nc[0], decoder_nc[4])
        self.outconv = outblock(decoder_nc[4], decoder_nc[5], stride=1)

        self.stn = Dense3DSpatialTransformer()

    def forward(self, input):
        # input shape: [B, 2, 128, 128] -> 2 stands for: 1 channel image + 1 channel code/noise
        x1 = self.inconv(input)
        x2 = self.downconv1(x1)
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)
        x5 = self.downconv4(x4)

        x = self.upconv1(x5, x4)
        x = self.upconv2(x, x3)
        x = self.upconv3(x, x2)
        x = self.keepblock(x)
        x = self.upconv4(x, x1)

        flow = self.outconv(x)
        mov = (input[:, :1] + 1) / 2.0  # bring the moving image to [0,1]

        out = self.stn(mov, flow)
        return out, flow
