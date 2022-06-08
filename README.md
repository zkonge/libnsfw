# LibNSFW
Minimal HTTP server provides nsfw image detection.

以 HTTP 方式提供的轻量级 NSFW 图片鉴别服务

## Usage
```
./libnsfw bind_addr nsfw_model worker_thread
```

### Example
```
./libnsfw 127.0.0.1:8000 ./nsfw.onnx 4
```

### HTTP Request
Just POST form-data with `image` key.
### HTTP Response
```
{
    "drawings": 0.5251695,
    "hentai": 0.47225672,
    "neutral": 0.0011893457,
    "porn": 0.0011269405,
    "sexy": 0.00025754774
}
```

## Thanks
[GantMan/nsfw_model](https://github.com/GantMan/nsfw_model) and [infinitered/nsfwjs](https://github.com/infinitered/nsfwjs)

## License
MIT