# 生成完整的HTML文档结构
输入"!" 获得
```html
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>Document</title>

</head>

<body>

</body>

</html>
```

# 常用文本标签
## 六个等级的标题
```html
    <h1>一级标题标签</h1>

    <h2>二级标题标签</h2>

    <h3>三级标题标签</h3>

    <h4>四级标题标签</h4>

    <h5>五级标题标签</h5>

    <h6>六级标题标签</h6>
```

## 段落标签和文本样式
```html
   <p>段落标签</p>

    <p>更改文本样式：<b>字体加粗</b><strong>加粗</strong></p>

    <p>更改文本样式：<i>字体倾斜</i><em></em></p>

    <p>更改文本样式：<u>下划线</u></p>

    <p>更改文本样式：<del>删除线</del>或者 <s>删除</s></p>
```

## 列表标签
- 无序列表
```html
    <ul>

        <li>无序列表元素 1 </li>

        <li>无序列表元素 2 </li>

        <li>无序列表元素 3 </li>

    </ul>
```
- 有序列表
```html
    <ol>

        <li>有序列表元素 1 </li>

        <li>有序列表元素 2 </li>

        <li>有序列表元素 3 </li>

    </ol>

```
## 表格标签
tr->table row
td->table data
th->table header
```html
<table border="1">

        <tr>

            <th>表头1</th>

            <th>表头2</th>

            <th>表头3</th>

        </tr>

        <tr>

            <td>数据1</td>

            <td>数据2</td>

            <td>数据3</td>

        </tr>

        <tr>

            <td>数据4</td>

            <td>数据5</td>

            <td>数据6</td>

        </tr>

    </table border="1">
```

# HTML属性
- 基本语法：`<开始标签 属性名="熟悉值">`
- 每个HTML元素可以具有不同的属性
```html
<p id="describe"class="section">这是一个段落标签</p>
<a href="https://www.baidu.com">这是一个超链接</a>
```
- 属性名称不区分大小写，属性值对大小写敏感
```html
<imggsrc="example.jpg" alt="">
<imgSRC="example.jpg" alt="">
<img Src="EXAMPLE.JPG" alt="">
<！--前两者相同，第三个与前两个不一样-->
```
## 适用于大多数HTML元素的属性
|  属性   |              描述              |
| :---: | :--------------------------: |
| class | 为HTML元素定义一个或多个类名(类名从样式文件引I入) |
|  id   |          定义元素唯一的id           |
| style |          规定元素的行内样式           |
例如：
```html
<h1 id="title"></h1>
<div class="nav-bar"></div>
<h2 class="nav-bar"></h2>
```

## 超链接标签和图标签
a: 超链接
img: 图标签
br: 换行符
hr: 分割线
```html
    <a href="https://mp.weixin.qq.com/s/K8FJtkash1ci2HPOXgh07A">机智流推文</a>

    <br>
    

    <a href="https://mp.weixin.qq.com/s/K8FJtkash1ci2HPOXgh07A" target="_blank">机智流推文在新窗口打开</a>

    <hr>

    <img src="https://www.baidu.com/img/bd_logo1.png" alt="百度logo" width="100" height="100">

    <!-- 可以是相对路径，也可以是绝对路径，也可以是网络路径，alt是代替字样，当图片无法显示时，会显示这个字样 -->

    <!-- width="100" height="100"可以不写，默认图像大小 -->
```

## HTML区块-块元素于行内元素
### 块元素(block)
块级元素通常用于组织和布局页面的主要结构和内容，例如段落、标题、列表、表格等。它们用于创建页面的主要部分，将内容分隔成逻辑块。
- 块级元素通常会从新行开始，并占据整行的宽度，因此它们会在页面上呈现为一块独立的内容块.
- 可以包含其他块级元素和行内元素。
- 常见的块级元素包括$<div>，<p>，<h1>到<h6>，<ul>，<ol>，<li>，<table>，<form>$等。


```html
    <div class="nav">

        <h1>div标签</h1>

        <p>div标签是一个块级元素，可以用来包裹一组元素，用来组织页面结构</p>

        <a href="#">链接 1</a>

        <a href="#">链接 2</a>

        <a href="#">链接 3</a>

    </div>

    <!-- .nav会复制一份样式

    #nav是附带id的

    或者div.nav -->

    <div class="content">

        <h1>文章标题</h1>

        <p>文章内容</p>

        <p>文章内容</p>

        <p>文章内容</p>

        <p>文章内容</p>

  

    </div>

    <span>span标签1</span>

    <span>span标签2</span>

    <span>span标签3</span>

    <hr>

    <span>链接点击这里<a href="#">链接</a></span>
```

# HTML表单

使用form包裹
form 里面有action->action是提交的地址 一般填api
## 输入
input
### 普通输入
```html
        <span>用户名：</span>

        <input type="text" placeholder="出现，填写后消失">

        <br>

        <span>密码：</span>

        <input type="text" value="自动填写">
```
此处区分两种
- placeholder: 出现 点击填写后消失
- value: 相当于默认值

### 密码输入
type="password"
使得字输入为点

### 选项
type="radio"
#### 单选
需要规定name
```html
        <label for="">性别</label>

        <input type="radio" name="gender">选项1 加name属性，只能单选

        <input type="radio" name="gender">选项2

        <input type="radio" name="gender">选项3
```
#### 不定选
不规定name
```html
        <label for="">性别</label>

        <input type="radio">选项1 不加name属性，可以多选

        <input type="radio">选项2

        <input type="radio">选项3
```
#### 多选
type="checkbox"
```html
        <label for="">多选 爱好</label>

        <input type="checkbox" name="hobby">唱

        <input type="checkbox" name="hobby">跳

        <input type="checkbox" name="hobby">rap

        <input type="checkbox" name="hobby">篮球
```
### 提交
type="submit"
```html
<input type="submit" name="" id="" value="上传（不写默认是提交）">
```