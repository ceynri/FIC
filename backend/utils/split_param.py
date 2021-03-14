import torch


def split_param(codec):
    torch.save(codec.encoder.state_dict(), './data/encoder_param.pth')
    torch.save(codec.decoder.state_dict(), './data/decoder_param.pth')
    torch.save(
        codec.entropy_bottleneck.state_dict(),
        './data/entropy_bottleneck_param.pth'
    )
    return
