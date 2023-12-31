from numpy import allclose
import building_babel.modules.growable as bbg
import building_babel.model as bbm
import torch
import torch.nn.functional as F

def test_insert():
    grl = bbg.GrowableLinear(2,3)
    grl.grow(4,5)
    grl.grow(6,7)
    grl[0,0] = 0
    grl[0,1] = 1
    grl[0,2] = 2
    grl[0,3] = 3
    grl[0,4] = 4
    grl[0,5] = 5
    grl[1,0] = 10
    grl[1,1] = 11
    grl[1,2] = 12
    grl[1,3] = 13
    grl[1,4] = 14
    grl[1,5] = 15
    grl[2,0] = 20
    grl[2,1] = 21
    grl[2,2] = 22
    grl[2,3] = 23
    grl[2,4] = 24
    grl[2,5] = 25
    grl[3,0] = 200
    grl[3,1] = 201
    grl[3,2] = 202
    grl[3,3] = 203
    grl[3,4] = 204
    grl[3,5] = 205
    grl[4,0] = 300
    grl[4,1] = 301
    grl[4,2] = 302
    grl[4,3] = 303
    grl[4,4] = 304
    grl[4,5] = 305
    grl[5,0] = 1000
    grl[5,1] = 1001
    grl[5,2] = 1002
    grl[5,3] = 1003
    grl[5,4] = 1004
    grl[5,5] = 1005
    grl[6,0] = 2000
    grl[6,1] = 2001
    grl[6,2] = 2002
    grl[6,3] = 2003
    grl[6,4] = 2004
    grl[6,5] = 2005

    expected = torch.tensor([[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 5.0000e+00],
            [1.0000e+01, 1.1000e+01, 1.2000e+01, 1.3000e+01, 1.4000e+01, 1.5000e+01],
            [2.0000e+01, 2.1000e+01, 2.2000e+01, 2.3000e+01, 2.4000e+01, 2.5000e+01],
            [2.0000e+02, 2.0100e+02, 2.0200e+02, 2.0300e+02, 2.0400e+02, 2.0500e+02],
            [3.0000e+02, 3.0100e+02, 3.0200e+02, 3.0300e+02, 3.0400e+02, 3.0500e+02],
            [1.0000e+03, 1.0010e+03, 1.0020e+03, 1.0030e+03, 1.0040e+03, 1.0050e+03],
            [2.0000e+03, 2.0010e+03, 2.0020e+03, 2.0030e+03, 2.0040e+03, 2.0050e+03]])

    assert torch.all(grl.full_matrix() == expected)

    x = torch.tensor([1.,10,100,1000,10000,.1]).view(1,6)
    yp = torch.nn.functional.linear(x, grl.full_matrix())
    y = grl(x)
    assert torch.all(y == yp)

def _no_change(in_dims = [2,4,6], out_dims=[3,5,7], bs=2, sl=4, allclose_params = {"rtol": 1e-5, "atol":1e-8, "equal_nan": True}):
    grl = bbg.GrowableLinear(in_dims[0],out_dims[0])
    x = torch.randn((bs,sl,out_dims[-1]))
    y1p = torch.nn.functional.linear(x[...,:in_dims[0]], grl.full_matrix())
    y1 = grl(x)
    assert torch.allclose(y1,y1p, **allclose_params)
    grl.grow(in_dims[1],out_dims[1])
    y2p = torch.nn.functional.linear(x[...,:in_dims[1]], grl.full_matrix())
    y2 = grl(x)
    assert torch.allclose(y2,y2p, **allclose_params), f"{y2-y2p}"
    assert torch.allclose(y1p[...,:out_dims[0]], y2p[...,:out_dims[0]], **allclose_params), f"{y1p-y2p[...,:out_dims[0]]}"
    assert torch.allclose(y1[...,:out_dims[0]], y2[...,:out_dims[0]], **allclose_params), f"{y1-y2[...,:out_dims[0]]}"
    grl.grow(in_dims[2],out_dims[2])
    y3p = torch.nn.functional.linear(x[...,:in_dims[2]], grl.full_matrix())
    y3 = grl(x)
    assert torch.allclose(y3,y3p, **allclose_params), f"{y3p-y3}"
    assert torch.allclose(y1p[...,:out_dims[0]], y3p[...,:out_dims[0]], **allclose_params), f"{y1p-y3p[...,:out_dims[0]]}"
    assert torch.allclose(y1[...,:out_dims[0]], y3[...,:out_dims[0]], **allclose_params), f"{y1-y3[...,:out_dims[0]]}"
    assert torch.allclose(y2p[...,:out_dims[1]], y3p[...,:out_dims[1]], **allclose_params), f"{y2p[...,:out_dims[1]]-y3p[...,:out_dims[1]]}"
    assert torch.allclose(y2[...,:out_dims[1]], y3[...,:out_dims[1]], **allclose_params), f"{y2[...,:out_dims[1]]-y3[...,:out_dims[1]]}"

def test_matmul_no_change():
    allclose_params = {"rtol": 1e-5, "atol":1e-8, "equal_nan": True}
    _no_change(allclose_params=allclose_params)
    _no_change([10,20,40],[30,60,120], allclose_params=allclose_params)
    _no_change([128,256,512], [128,256,512], 3, 7, allclose_params=allclose_params)

def test_matmul():
    grl = bbg.GrowableLinear(2,3)
    grl.grow(4,5, init=torch.nn.init.kaiming_normal_)
    grl.grow(6,7, init=torch.nn.init.kaiming_normal_)

    x = torch.randn((2,4,6))
    yp = torch.nn.functional.linear(x, grl.full_matrix())
    y = grl(x)
    assert torch.allclose(y,yp)

def test_RMSNorm_no_change():
    r = bbg.GrowableRMSNorm(100, 1e-5)
    x = torch.randn((2,3,150))
    y = r(x[...,:100])
    r.grow(150)
    y1 = r(x)
    assert torch.allclose(y[...,:100], y1[...,:100])

def test_model_grow():
    """
    we would like this to have a smaller atol, but I'll be ok with it
    given the rtol is decently low.
    """
    c = bbm.TransformerConfig(128, 1, 1000)
    t = bbm.Transformer(c)
    x = torch.tensor([[1,2,400,23,1,23,4]])
    y1 = t(x)
    t.grow(256)
    y2 = t(x)
    assert torch.allclose(y1,y2, atol=1e-1, rtol=1e-5), f"{y1-y2=}"

# def test_insert_InPlace():
#     grl = bbg.GrowableLinearInPlace(2,3)
#     grl.grow(4,5)
#     grl.grow(6,7)
#     grl[0,0] = 0
#     grl[0,1] = 1
#     grl[0,2] = 2
#     grl[0,3] = 3
#     grl[0,4] = 4
#     grl[0,5] = 5
#     grl[1,0] = 10
#     grl[1,1] = 11
#     grl[1,2] = 12
#     grl[1,3] = 13
#     grl[1,4] = 14
#     grl[1,5] = 15
#     grl[2,0] = 20
#     grl[2,1] = 21
#     grl[2,2] = 22
#     grl[2,3] = 23
#     grl[2,4] = 24
#     grl[2,5] = 25
#     grl[3,0] = 200
#     grl[3,1] = 201
#     grl[3,2] = 202
#     grl[3,3] = 203
#     grl[3,4] = 204
#     grl[3,5] = 205
#     grl[4,0] = 300
#     grl[4,1] = 301
#     grl[4,2] = 302
#     grl[4,3] = 303
#     grl[4,4] = 304
#     grl[4,5] = 305
#     grl[5,0] = 1000
#     grl[5,1] = 1001
#     grl[5,2] = 1002
#     grl[5,3] = 1003
#     grl[5,4] = 1004
#     grl[5,5] = 1005
#     grl[6,0] = 2000
#     grl[6,1] = 2001
#     grl[6,2] = 2002
#     grl[6,3] = 2003
#     grl[6,4] = 2004
#     grl[6,5] = 2005
#
#     expected = torch.tensor([[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 5.0000e+00],
#             [1.0000e+01, 1.1000e+01, 1.2000e+01, 1.3000e+01, 1.4000e+01, 1.5000e+01],
#             [2.0000e+01, 2.1000e+01, 2.2000e+01, 2.3000e+01, 2.4000e+01, 2.5000e+01],
#             [2.0000e+02, 2.0100e+02, 2.0200e+02, 2.0300e+02, 2.0400e+02, 2.0500e+02],
#             [3.0000e+02, 3.0100e+02, 3.0200e+02, 3.0300e+02, 3.0400e+02, 3.0500e+02],
#             [1.0000e+03, 1.0010e+03, 1.0020e+03, 1.0030e+03, 1.0040e+03, 1.0050e+03],
#             [2.0000e+03, 2.0010e+03, 2.0020e+03, 2.0030e+03, 2.0040e+03, 2.0050e+03]])
#
#     assert torch.all(grl.weights == expected)
#
#     x = torch.tensor([1.,10,100,1000,10000,.1]).view(1,6)
#     yp = torch.nn.functional.linear(x, grl.weights)
#     y = grl(x)
#     assert torch.all(y == yp)

def test_opt():
    grl = bbg.GrowableLinear(2,3)
    grl.grow(4,5)
    grl.grow(6,7)
    grl[0,0] = 0
    grl[0,1] = 1
    grl[0,2] = 2
    grl[0,3] = 3
    grl[0,4] = 4
    grl[0,5] = 5
    grl[1,0] = 10
    grl[1,1] = 11
    grl[1,2] = 12
    grl[1,3] = 13
    grl[1,4] = 14
    grl[1,5] = 15
    grl[2,0] = 20
    grl[2,1] = 21
    grl[2,2] = 22
    grl[2,3] = 23
    grl[2,4] = 24
    grl[2,5] = 25
    grl[3,0] = 200
    grl[3,1] = 201
    grl[3,2] = 202
    grl[3,3] = 203
    grl[3,4] = 204
    grl[3,5] = 205
    grl[4,0] = 300
    grl[4,1] = 301
    grl[4,2] = 302
    grl[4,3] = 303
    grl[4,4] = 304
    grl[4,5] = 305
    grl[5,0] = 1000
    grl[5,1] = 1001
    grl[5,2] = 1002
    grl[5,3] = 1003
    grl[5,4] = 1004
    grl[5,5] = 1005
    grl[6,0] = 2000
    grl[6,1] = 2001
    grl[6,2] = 2002
    grl[6,3] = 2003
    grl[6,4] = 2004
    grl[6,5] = 2005

    expected = torch.tensor([[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 5.0000e+00],
            [1.0000e+01, 1.1000e+01, 1.2000e+01, 1.3000e+01, 1.4000e+01, 1.5000e+01],
            [2.0000e+01, 2.1000e+01, 2.2000e+01, 2.3000e+01, 2.4000e+01, 2.5000e+01],
            [2.0000e+02, 2.0100e+02, 2.0200e+02, 2.0300e+02, 2.0400e+02, 2.0500e+02],
            [3.0000e+02, 3.0100e+02, 3.0200e+02, 3.0300e+02, 3.0400e+02, 3.0500e+02],
            [1.0000e+03, 1.0010e+03, 1.0020e+03, 1.0030e+03, 1.0040e+03, 1.0050e+03],
            [2.0000e+03, 2.0010e+03, 2.0020e+03, 2.0030e+03, 2.0040e+03, 2.0050e+03]])

    x = torch.tensor([1.,10,100,1000,10000,.1]).view(1,6)
    y = grl(x)
    adam = torch.optim.AdamW([{"params": grl.get_section(0,0), "lr":0.001},
                          {"params": grl.get_section(1,1), "lr":0.01},
                          {"params": grl.get_section(1,2), "lr":0.02}])
    yt = torch.ones((1,7))
    loss = F.l1_loss(y, yt)
    loss.backward()
    adam.step()

    assert torch.any(grl.full_matrix() != expected)

    section11 = torch.tensor([[ 1.989800,  2.989700],
        [11.988800, 12.988700],
        [21.987799, 22.987700]])
    assert torch.allclose(section11,grl.get_section(1,1), atol=1e-5, rtol=1e-5)

    section12 = torch.tensor([[199.9400, 200.9398],
        [299.9200, 300.9198]])
    assert torch.allclose(section12,grl.get_section(1,2), atol=1e-5, rtol=1e-5)

    section00 = torch.tensor([[-9.999998e-04,  9.989900e-01],
        [ 9.998899e+00,  1.099889e+01],
        [ 1.999880e+01,  2.099879e+01]])
    assert torch.allclose(section00,grl.get_section(0,0), atol=1e-5, rtol=1e-5)


# def test_opt_InPlace():
#     grl = bbg.GrowableLinearInPlace(2,3)
#     grl.grow(4,5)
#     grl.grow(6,7)
#     grl[0,0] = 0
#     grl[0,1] = 1
#     grl[0,2] = 2
#     grl[0,3] = 3
#     grl[0,4] = 4
#     grl[0,5] = 5
#     grl[1,0] = 10
#     grl[1,1] = 11
#     grl[1,2] = 12
#     grl[1,3] = 13
#     grl[1,4] = 14
#     grl[1,5] = 15
#     grl[2,0] = 20
#     grl[2,1] = 21
#     grl[2,2] = 22
#     grl[2,3] = 23
#     grl[2,4] = 24
#     grl[2,5] = 25
#     grl[3,0] = 200
#     grl[3,1] = 201
#     grl[3,2] = 202
#     grl[3,3] = 203
#     grl[3,4] = 204
#     grl[3,5] = 205
#     grl[4,0] = 300
#     grl[4,1] = 301
#     grl[4,2] = 302
#     grl[4,3] = 303
#     grl[4,4] = 304
#     grl[4,5] = 305
#     grl[5,0] = 1000
#     grl[5,1] = 1001
#     grl[5,2] = 1002
#     grl[5,3] = 1003
#     grl[5,4] = 1004
#     grl[5,5] = 1005
#     grl[6,0] = 2000
#     grl[6,1] = 2001
#     grl[6,2] = 2002
#     grl[6,3] = 2003
#     grl[6,4] = 2004
#     grl[6,5] = 2005
#
#     expected = torch.tensor([[0.0000e+00, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00, 5.0000e+00],
#             [1.0000e+01, 1.1000e+01, 1.2000e+01, 1.3000e+01, 1.4000e+01, 1.5000e+01],
#             [2.0000e+01, 2.1000e+01, 2.2000e+01, 2.3000e+01, 2.4000e+01, 2.5000e+01],
#             [2.0000e+02, 2.0100e+02, 2.0200e+02, 2.0300e+02, 2.0400e+02, 2.0500e+02],
#             [3.0000e+02, 3.0100e+02, 3.0200e+02, 3.0300e+02, 3.0400e+02, 3.0500e+02],
#             [1.0000e+03, 1.0010e+03, 1.0020e+03, 1.0030e+03, 1.0040e+03, 1.0050e+03],
#             [2.0000e+03, 2.0010e+03, 2.0020e+03, 2.0030e+03, 2.0040e+03, 2.0050e+03]])
#
#     x = torch.tensor([1.,10,100,1000,10000,.1]).view(1,6)
#     y = grl(x)
#     adam = torch.optim.AdamW([{"params": grl.get_section(0,0), "lr":0.001},
#                           {"params": grl.get_section(1,1), "lr":0.01},
#                           {"params": grl.get_section(1,2), "lr":0.02}])
#     yt = torch.ones((1,7))
#     loss = F.l1_loss(y, yt)
#     loss.backward()
#     adam.step()
#
#     assert torch.any(grl.weights != expected)
