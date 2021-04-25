---
title: '深度学习算法服务的搭建与部署教程'
date: 2021-04-12
---

# 深度学习算法服务的搭建与部署教程

## 服务器预备环境

- Nginx
- Python 3.x
- Flask

## 请求代理逻辑

我们先纵览一下用户请求在服务器内的路由逻辑，从总体结构上对整个算法服务的结构有一个大概的认识。

服务器环境：

- Nginx 监听服务器 80 端口，扮演代理请求的角色
- Flask 监听服务器 xxxx 端口（举例为 1127 端口），接收 WEB 请求并调用对应的 Python 代码进行处理

### 静态资源请求

用户访问网站时，会请求服务器的对应的网页文件。Nginx 收到请求后，分析路由规则，找到指定目录下的文件并返回给用户。

![请求网页.png](https://i.loli.net/2021/04/11/AKWnOcv5fl1RCuk.png)

一个例子：

![请求网页例子.png](https://i.loli.net/2021/04/11/jTChRVMIFA6yOce.png)

### 接口请求

当用户点击网页上的特定按钮时（例如上传图片执行压缩的功能），会发起 POST 请求，携带上传文件（如果有）请求指定的接口。Nginx 收到请求后，分析匹配到路由规则为“代理转发”，则直接将请求转发给 Flask 服务。

![请求算法接口.png](https://i.loli.net/2021/04/11/qVFvaOnuoH4NDbr.png)

可以看出 Flask 服务并没有直接与用户请求通信，中间加了一层 Nginx 作为代理。

一个例子：

![请求算法接口例子.png](https://i.loli.net/2021/04/11/SDXuygwiNZV6L1z.png)

## 部署配置

### 使用 Flask 编写算法服务

Flask 是一个由 Python 编写的轻量级 WEB 框架，API 简单易上手。

一个最简单的响应 `Hello World!` 的服务代码如下所示：

```python
# hello.py

from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'
```

Flask 框架代码的特点是使用了装饰器语法，`@app.route('/')` 下的 `hello_world()` 即代表了该装饰器所修饰的函数。该函数会在用户访问网站的根目录时触发执行（即 `return 'Hello world!'`）。

作为一个简单的算法服务，我们只需要了解这点，然后就可以尝试编写它了。

想要运行 flask 服务，即在项目目录下的命令行输入以下指令即可：

```bash
export FLASK_APP=hello.py
flask run -p 7211
```

其中 `-p` 用来指定 flask 服务所监听的端口号，任意未被占用的端口都可以。

如果要启用调试模式，则在 run 之前需要多加一个环境变量：

```bash
export FLASK_APP=hello.py
export FLASK_ENV=development
flask run -p 7211
```

添加后，启用的服务会随着代码的更新而自动重启，便于开发调试。

在路由所触发的函数中添加调用算法的代码，并在算法处理结束后将结果进行 `return`，就实现了算法的 WEB 服务化。

### 接收文件

对于深度学习而言，模型往往需要接受数据的输入。Flask 提供了接受文件的 API，例子如下：

```python
from io import BytesIO
from PIL import Image
from flask import request

@app.route('/uploads', methods=['POST'])
def upload_file():
    '''保存到指定的服务器目录下'''

    file = request.files['the_file']
    file.save(f'/path/to/your/uploads/{file.filename}')


@app.route('/process', methods=['POST'])
def process_file():
    '''或者直接使用，以图片文件转为 PIL.Image 为例'''

    file = request.files['the_file']
    data = file.read()
    stream = BytesIO(data)
    img = Image.open(stream)
    # ...
```

其中，`the_file` 是与前端网页所约定好的字段名，可以取任意其他指定好的名字。

### 返回结果

一般而言，当深度学习算法处理完得到的结果，如果是一个数据文件（如图片、二进制文件），我们不直接以二进制流的形式直接返回请求。

我们可以将处理好的结果文件保存于指定的服务器文件夹下，然后该接口返回可以路由到该文件的 url 地址，由前端来主动请求该文件。

以图片为例：

```python
from flask import jsonify

@app.route('/process', methods=['POST'])
def process_file():
    # 获取 input_img
    # ...

    result_img = Net(input_img)
    result_img.save('/path/to/your/assets/test.jpg')
    return jsonify({
        'result': '/assets/test.jpg',
    })
```

这里的路径需要由 Nginx 进行代理，使得访问 `http://example.com/assets/test.jpg` 时能够获取到服务器上路径为 `/path/to/your/assets/test.jpg` 的文件。

`return` 返回的字典使用 `jsonify()` 方法进行包裹，使其以前端易接受的 json 的格式返回，便于后续的处理。

> 更多的 Flask API 与知识点，可以参考 [Flask 官方教程](https://dormousehole.readthedocs.io/en/latest/quickstart.html)。

### 使用 Nginx 代理静态资源与算法服务

Nginx 是目前使用非常广泛的高性能 HTTP 和反向代理 web 服务器。

在这里，我们使用 Nginx **代理静态资源**，使得用户可以正常浏览网站。

我们还使用 Nginx 对 Flask 所起的算法服务做一个**反向代理**，使用户在请求接口时，隐藏 Flask 服务的实际端口，同时避免出现跨域等问题。

Nginx 启动方法根据安装方式等实际情况有所区别，具体以安装教程为准。

Nginx 的配置文件为 `nginx.conf`，具体位置也因 Nginx 的安装方式而有不同，典型的位置例如`/etc/nginx/nginx.conf`、`/usr/local/bin/nginx/conf/nginx.conf`。

配置规则可以参考网络教程（例如[这个](https://blog.51cto.com/u_13363488/2349546)），这里简单举一下静态资源代理和接口服务反向代理的相关配置方式的例子：

```conf
# nginx.conf

http {
    # ...

    include       mime.types;
    default_type  application/octet-stream;

    server {
        # ...

        listen       80;
        server_name  localhost;

        # location 语法： location [=|~|~*|^~] /uri/ { ... }
        # 默认路由，优先级低于下面的匹配规则
        location / {
            # 匹配任意路由，定位到对应的/path/to/your/website/下的路径中
            alias  /path/to/your/website/;

            # 例如：http://127.0.0.1/demo/index.html
            # 对应的服务器路径为 /path/to/your/website/demo/index.html
        }

        # 匹配以 /assets/ 开头的路由
        location ^~ /assets/ {
            # 对应的服务器路径为 /path/to/assets/xxx
            # 注意：不是 /assets/path/to/assets/xxx，前面不会保留 /assets/
            alias  /path/to/assets/;
        }

        # 匹配以 /api/ 开头的路由
        location ^~ /api/ {
            # 代理转发到本地的 1127 端口上
            # 转发后，路由路径直接取后面的部分，而不会以 /api/ 作为路由的开头
            proxy_pass http://127.0.0.1:1127/;
        }
    }
}
```

> 实际上，Flask 也可以做网页资源的代理，即监听路由返回对应的静态资源文件即可，从而无需使用 Nginx。
>
> 此处使用 Nginx 作为代理，是为了更强的静态资源请求响应能力和更好的服务拓展性。

### 前端上传文件

前端上传文件存在多种方法，简单的使用原生HTML标签 form 与 input 实现，还有使用 formData 或者 fileReader 实现的方法。

这里演示一个基于 Vue 的使用 FormData 上传多个文件的方法：

```html
<template>
    <input type="file" multiple @change="getFiles" />
</template>

<script>
import Axios from 'axios';

export default {
    methods: {
        /**
         * 从原生Input文件中获取files
         */
        getFiles(e) {
            this.uploads(e.target.files);
        },
        /**
         * 携带 files 请求 /api/compress 接口
         */
        uploads(files) {
            const data = new FormData();
            files.forEach((file) => {
                data.append('the_file', file);
            });
            const res = await Axios.post('/api/compress', data, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });
            return res.data;
        },
    }
}
</script>
```

如果使用的是原生 HTML + JavaScript，则对应的代码大致如下（未经验证）：

```html
<body>
    <!-- ... -->
    <input type="file" multiple @change="getFiles" id="fileInput" />
    <!-- ... -->
</body>

<script>
    const fileInput = document.querySelector('#fileInput');
    fileInput.addEventListener('change', (e) => {
        // 获取文件
        const files = e.target.files;
        // 转换为 FormData
        const data = new FormData();
        files.forEach((file) => {
            data.append('the_file', file);
        });

        // 创建请求
        const req = new XMLHttpRequest();
        // post 请求
        req.open('POST', '/api/compress');
        // 设置文件上传的请求头
        req.setRequestHeader('Content-Type','multipart/form-data');
        // 携带文件数据
        req.send(data);
        // 设置请求监听
        req.onreadystatechange = () => {
            if (req.readyState == 4 && req.status == 200) {
                const result = req.responseText;
                // 使用返回的结果
                // console.log(result);
                // ...
            }
        };
    })
</script>
```
