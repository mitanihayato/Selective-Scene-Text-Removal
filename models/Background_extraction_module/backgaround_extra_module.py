import torch

temp_shape = (0,0)

def calc_padding(h, w, k, s):
    
    h_pad = (((h-1)*s) + k - h)//2 
    w_pad = (((w-1)*s) + k - w)//2
    
    return (h_pad, w_pad)

def calc_inv_padding(h, w, k, s):
    h_pad = (k-h + ((h-1)*s))//2
    w_pad = (k-w + ((w-1)*s))//2
    
    return (h_pad, w_pad)


class Conv_bn_block(torch.nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._bn = torch.nn.BatchNorm2d(kwargs['out_channels'])
        
    def forward(self, input):
        return torch.nn.functional.leaky_relu(self._bn(self._conv(input)),negative_slope=0.2)

class Res_block(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
            
        self._conv1 = torch.nn.Conv2d(in_channels, in_channels//4, kernel_size = 1, stride =1)
        self._conv2 = torch.nn.Conv2d(in_channels//4, in_channels//4, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv3 = torch.nn.Conv2d(in_channels//4, in_channels, kernel_size = 1, stride=1)
        
        self._bn = torch.nn.BatchNorm2d(in_channels)
       
    def forward(self, x):
        
        xin = x
        x = torch.nn.functional.leaky_relu(self._conv1(x),negative_slope=0.2)
        x = torch.nn.functional.leaky_relu(self._conv2(x),negative_slope=0.2)
        x = self._conv3(x)
        x = torch.add(xin, x)
        x = torch.nn.functional.leaky_relu(self._bn(x),negative_slope=0.2)
        
        return x

class encoder_net(torch.nn.Module):
    
    def __init__(self, in_channels, get_feature_map = False):
        super().__init__()
            
        self.cnum = 64
        self.get_feature_map = get_feature_map
        self._conv1_1 = Conv_bn_block(in_channels = in_channels, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv1_2 = Conv_bn_block(in_channels =self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #--------------------------
        self._pool1 = torch.nn.Conv2d(in_channels = self.cnum, out_channels = 2 * self.cnum, kernel_size = 2, stride = 2, padding = 0)
        
        self._conv2_1 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv2_2 = Conv_bn_block(in_channels = 2* self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #---------------------------
        self._pool2 = torch.nn.Conv2d(in_channels = 2*self.cnum, out_channels = 4* self.cnum, kernel_size = 2, stride = 2, padding = 0)
        
        self._conv3_1 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv3_2 = Conv_bn_block(in_channels = 4* self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #---------------------------
        self._pool3 = torch.nn.Conv2d(in_channels = 4*self.cnum, out_channels = 8* self.cnum, kernel_size = 2, stride = 2, padding = 0)
        
        self._conv4_1 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv4_2 = Conv_bn_block(in_channels = 8* self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        
    def forward(self, x):
                
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        
        x = torch.nn.functional.leaky_relu(self._pool1(x),negative_slope=0.2)
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        
        f1 = x
        
        x = torch.nn.functional.leaky_relu(self._pool2(x),negative_slope=0.2)
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        
        f2 = x
        
        x = torch.nn.functional.leaky_relu(self._pool3(x),negative_slope=0.2)
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        
        
        if self.get_feature_map:
            return x, [f2, f1]
        
        else:
            return x
        
        
class build_res_block(torch.nn.Module):
    
    def __init__(self, in_channels):
        
        super().__init__()
        
        self._block1 = Res_block(in_channels)
        self._block2 = Res_block(in_channels)
        self._block3 = Res_block(in_channels)
        self._block4 = Res_block(in_channels)
        
    def forward(self, x):
        
        x = self._block1(x)
        x = self._block2(x)
        x = self._block3(x)
        x = self._block4(x)
        
        return x
    
    
class decoder_net(torch.nn.Module):
    
    def __init__(self, in_channels, get_feature_map = False, mt =1, fn_mt=1):
        super().__init__()
        
        self.cnum = 64
       
        self.get_feature_map = get_feature_map
        
        self._conv1_1 = Conv_bn_block(in_channels = fn_mt*in_channels , out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1) 

        self._conv1_2 = Conv_bn_block(in_channels = 8*self.cnum, out_channels = 8*self.cnum, kernel_size = 3, stride =1, padding = 1)
        
        #-----------------
        self._deconv1 = torch.nn.ConvTranspose2d(8*self.cnum, 4*self.cnum, kernel_size = 2, stride = 2, padding = 0)
        
        self._conv2_1 = Conv_bn_block(in_channels = fn_mt*mt*4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv2_2 = Conv_bn_block(in_channels = 4*self.cnum, out_channels = 4*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        #-----------------
        self._deconv2 = torch.nn.ConvTranspose2d(4*self.cnum, 2*self.cnum, kernel_size =2 , stride = 2, padding = 0)
        
        self._conv3_1 = Conv_bn_block(in_channels = fn_mt*mt*2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        self._conv3_2 = Conv_bn_block(in_channels = 2*self.cnum, out_channels = 2*self.cnum, kernel_size = 3, stride = 1, padding = 1)
       
        #----------------
        self._deconv3 = torch.nn.ConvTranspose2d(2*self.cnum, self.cnum, kernel_size =2 , stride = 2, padding = 0)
        
        self._conv4_1 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        self._conv4_2 = Conv_bn_block(in_channels = self.cnum, out_channels = self.cnum, kernel_size = 3, stride = 1, padding = 1)
        
        
    def forward(self, x, fuse = None):
        
        
        if fuse and fuse[0] is not None:
            x = torch.cat((x, fuse[0]), dim = 1)
            
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        f1 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self._deconv1(x), negative_slope=0.2)
       
        if fuse and fuse[1] is not None:
            x = torch.cat((x, fuse[1]), dim = 1)
            
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        f2 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self._deconv2(x), negative_slope=0.2)
        if fuse and fuse[2] is not None:
            x = torch.cat((x, fuse[2]), dim = 1)
        
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        f3 = x
        
        #----------
        
        x = torch.nn.functional.leaky_relu(self._deconv3(x), negative_slope=0.2)
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        
        if self.get_feature_map:
            return x, [f1, f2, f3]        
        
        else:
            return x
                                                  

class background_extraction_module(torch.nn.Module):
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.cnum = 64
        self._encoder = encoder_net(in_channels, get_feature_map = True)
        self._res = build_res_block(8*self.cnum)
        
        self._decoder = decoder_net(8*self.cnum,  get_feature_map = True, mt=2)
        self._out = torch.nn.Conv2d(self.cnum, 3, kernel_size = 3, stride = 1, padding = 1)
        
    def forward(self, x):
        
        x, f_encoder = self._encoder(x)
        x = self._res(x)

        x, fs = self._decoder(x, fuse = [None] + f_encoder)
        
        x = self._out(x)
        
        return x
