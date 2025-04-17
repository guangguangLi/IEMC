
__all__ = ['UNext']

class ViLLayer(nn.Module):
    def __init__(self, dim, out_dim, d_state = 16, d_conv = 4, expand = 2):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.proj = nn.Linear(dim, out_dim)
        self.vil = ViLBlock(
            dim= self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )
    
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_vil = self.vil(x_flat)
        
        out = self.proj(x_vil)
        
        out = out.transpose(-1, -2).reshape(B, self.out_dim, *img_dims)
        #out = x + out
        return out
        

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W      
        

        
        
class Curvature(torch.nn.Module):
    def __init__(self, ratio):
        super(Curvature, self).__init__()
        weights = torch.tensor([[[[-1/16, 5/16, -1/16], [5/16, -1, 5/16], [-1/16, 5/16, -1/16]]]])
        self.weight = torch.nn.Parameter(weights).cuda()
        self.ratio = ratio
 
    def forward(self, x):
        B, C, H, W = x.size()
        x_origin = x
        x = x.reshape(B*C,1,H,W)
        out = F.conv2d(x, self.weight)
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p=p.reshape(B, C)

        _, index = torch.topk(p, int(self.ratio*C), dim=1)
        selected = []
        for i in range(x_origin.shape[0]):
            selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)
        
        return selected

class Entropy_Hist(nn.Module):
    def __init__(self, ratio, win_w=3, win_h=3):
        super(Entropy_Hist, self).__init__()
        self.win_w = win_w
        self.win_h = win_h
        self.ratio = ratio

    def calcIJ_new(self, img_patch):
        total_p = img_patch.shape[-1] * img_patch.shape[-2]
        if total_p % 2 != 0:
            tem = torch.flatten(img_patch, start_dim=-2, end_dim=-1) 
            center_p = tem[:, :, :, int(total_p / 2)]
            mean_p = (torch.sum(tem, dim=-1) - center_p) / (total_p - 1)
            if torch.is_tensor(img_patch):
                return center_p * 100 + mean_p
            else:
                return (center_p, mean_p)
        else:
            print("modify patch size")

    def histc_fork(self,ij):
        BINS = 256
        B, C = ij.shape
        N = 16
        BB = B // N
        min_elem = ij.min()
        max_elem = ij.max()
        ij = ij.view(N, BB, C)

        def f(x):
            with torch.no_grad():
                res = []
                for e in x:
                    res.append(torch.histc(e, bins=BINS, min=min_elem, max=max_elem))
                return res
        futures : List[torch.jit.Future[torch.Tensor]] = []

        for i in range(N):
            futures.append(torch.jit.fork(f, ij[i]))

        results = []
        for future in futures:
            results += torch.jit.wait(future)
        with torch.no_grad():
            out = torch.stack(results)
        return out

    def forward(self, img):
        with torch.no_grad():
            B, C, H, W = img.shape
            ext_x = int(self.win_w / 2) 
            ext_y = int(self.win_h / 2)

            new_width = ext_x + W + ext_x 
            new_height = ext_y + H + ext_y
            
   
            nn_Unfold=nn.Unfold(kernel_size=(self.win_w,self.win_h),dilation=1,padding=ext_x,stride=1)

            x = nn_Unfold(img) # (B,C*K*K,L)
            x= x.view(B,C,3,3,-1).permute(0,1,4,2,3) 
            ij = self.calcIJ_new(x).reshape(B*C, -1) 
            #print(ij.shape)
            
            fij_packed = self.histc_fork(ij)
            p = fij_packed / (new_width * new_height)
            h_tem = -p * torch.log(torch.clamp(p, min=1e-40)) / math.log(2)

            a = torch.sum(h_tem, dim=1) 
            H = a.reshape(B,C) 

            _, index = torch.topk(H, int(self.ratio*C), dim=1) # Nx3 int(self.ratio*C)
        selected = []
        for i in range(img.shape[0]):
            selected.append(torch.index_select(img[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)
        
        return selected

       
        
class UNeXt(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP
    
    def __init__(self,  num_classes, input_channels=3, deep_supervision=False,img_size=224, patch_size=16, in_chans=3,  embed_dims=[ 128,160,256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        
        self.encoder1 = nn.Conv2d(3, 8, 3, stride=1, padding=1)  
        self.encoder2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)  
        self.encoder3 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder4 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.encoder5 = nn.Conv2d(64, 256, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(8)
        self.ebn2 = nn.BatchNorm2d(16)
        self.ebn3 = nn.BatchNorm2d(32)
        self.ebn4 = nn.BatchNorm2d(64)
        self.ebn5 = nn.BatchNorm2d(256)



        self.decoder1 =   nn.Conv2d(128, 64, 3, stride=1,padding=1)  
        self.decoder2 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)  
        self.decoder3 =   nn.Conv2d(32, 16, 3, stride=1, padding=1) 
        self.decoder4 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.decoder5 =   nn.Conv2d(8, 8, 3, stride=1, padding=1)
        

        self.dbn1 = nn.BatchNorm2d(64)
        self.dbn2 = nn.BatchNorm2d(32)
        self.dbn3 = nn.BatchNorm2d(16)
        self.dbn4 = nn.BatchNorm2d(8)
        
        self.final = nn.Conv2d(8, num_classes, 1, stride=1, padding=0)
        

    
        
        self.E5L2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, stride=1, padding=1),
        )
        
        self.E5L3 = nn.Sequential(
            nn.Conv2d(256, 128, 5, stride=1, padding=2),
        )
        
        self.E5L4 = nn.Sequential(
            nn.Conv2d(256, 128, 7, stride=1, padding=3),
        )
        
        self.E5L5 = nn.Sequential(
            nn.Conv2d(256, 128, 9, stride=1, padding=4),
        )
        
      
        self.ratio = [0.5, 0.5]
        self.ife1 = Curvature(self.ratio[0])
        self.ife3 = Entropy_Hist(self.ratio[1])


    def forward(self, x):
    
        B = x.shape[0]
        
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)),2,2))
        ### Stage 1
        t1 = out
        ### Stage 2
        
        out = out + self.E1(torch.cat([out, self.ife3(out)], dim=1))
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)),2,2))
        t2 = out
        ### Stage 3
        
        out = out + self.E2(torch.cat([out, self.ife3(out)], dim=1))
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)),2,2))
        t3 = out
        
        out = out + self.E3(torch.cat([out, self.ife3(out)], dim=1))
        out = F.relu(F.max_pool2d(self.ebn4(self.encoder4(out)),2,2))
        t4 = out
        
        out = out + self.E4(torch.cat([out, self.ife3(out)], dim=1))
        out = F.relu(F.max_pool2d(self.ebn5(self.encoder5(out)),2,2))
        
        
        F1 = F.relu(self.E5L2(out)) + F.relu(self.E5L3(out))
        F2 = F.relu(self.E5L4(out)) + F.relu(self.E5L5(out))
        k1 = torch.cat([F1, F2], dim=1)
        k1 = self.ife1(k1)
        out = k1 + F1 + F2
        
        

        
        
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t4)
        
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t3)

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t2)
        
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)),scale_factor=(2,2),mode ='bilinear'))
        out = torch.add(out,t1)
        

        out = F.relu(F.interpolate(self.decoder5(out),scale_factor=(2,2),mode ='bilinear'))
        
        return self.final(out)
    
        
            
        
            
        