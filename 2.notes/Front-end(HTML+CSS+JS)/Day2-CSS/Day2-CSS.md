# CSS 简介及导入方式
## CSS简介
CSS全名是`Cascading Style Sheets`，中文名层叠样式表。
用于定义网页样式和布局的样式表语言。
通过CSS，你可以指定页面中各个元素的颜色、字体、大小、间距、边框、背景等样式，从而实现更精确的页面设计。
## CSS语法
CSS通常由选择器、属性和属性值组成，多个规则可以组合在一起，以便同时应用多个样式
```css
选择器{
	属性1: 属性值1；
	属性2: 属性值2；
}
```

1. 选择器的声明中可以写无数条属性
2. 声明的每一行属性，都需要以英文分号结尾；
3. 声明中的所有属性和值都是以键值对这种形式出现的
### 示例
```css
/*这是一个p标签选择器*/
p{
	color: blue;
	font-size: 16px;
}
```

```html
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>CSS 导入方式</title>

    <style>

        p{

            color: red;

            font-size: 16px;

        }

    </style>

</head>

<body>

    <p>这是一个用了css样式的文本</p>

</body>

</html>
```

## CSS 三种导入方式
下面是三种常见的CSS导入方式：
1. 内联样式（Inline Styles）
2. 内部样式表（Internal Styles heet）
3. 外部样式表（External Stylesheet）
三种导入方式的优先级：内联样式>内部样式表>外部样式表

```css
h3{

    color: palegreen;

    font-size: 20px;

}
```

```html
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>CSS 导入方式</title>

    <link rel="stylesheet" href="./css/style.css">

    <style>

        p{

            color: red;

            font-size: 16px;

        }

    </style>

</head>

<body>

    <p>这是一个用了css内部样式的文本</p>

    <h1 style="

            color: blue;

            font-size: 24px;

            ">这是一个用了css样式的一级标题

    </h1>

  

    <h2 style="

            color: green;

            font-size: 20px;

            ">这是一个用了css样式的二级标题

    </h2>

  

    <h3>这是一个用了css样式的三级标题

    </h3>

</body>

</html>
```

# 选择器
选择器是CSS中的关键部分，它允许你针对特定元素或一组元素定义样式
- 元素选择器
- 类选择器
- ID选择器
- 通用选择器
- 子元素选择器
- 后代选择器（包含选择器）
- 并集选择器（兄弟选择器）
- 伪类选择器
```html
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>CSS 选择器</title>

    <style>

        /* 元素选择器 */

        h2{

            color: red;

  

        }

        /* 类选择器 */

        .hightlight

        {

            background-color: blue;

        }

        /* ID选择器 */

        #head{

            color: green;

        }

        /* 通用选择器 */

        *{

            font-family: kaiti;

        }

        /* 子元素选择器 */

        .father >.son{

            color: yellow;

        }

        /* 后代选择器 */

        .father p {

            color: pink;

            font-size: large;

        }

        /* 相邻兄弟选择器 */

        h3 + p{

            color: purple;

            background-color: lightblue;

        }

        /* 伪类选择器 */

        #element:hover{

            color: orange;

        }

        /* 选中第一个子元素 :first-child

        选中最后一个子元素 :last-child

        选中第一个子元素 :nth-child(1)

                       :active */

        /* 伪元素选择器

        ::after

        ::before          */

    </style>

</head>

<body>

    <h1>不同类型的CSS 选择器</h1>

  

    <h2>这是一个元素选择器</h2>

  

    <h3 class="hightlight">这是一个类选择器</h3>

    <h3 >这是另一个类选择器</h3>

  

    <h4 id="head">这是一个ID选择器</h4>

  

    <!-- .father -->

    <div class="father">

        <p class="son">这是一个子选择器</p>

        <div>

            <p class="grandson">这是一个后代选择器</p>

        </div>

    </div>

  

    <p>这是一个普通的p标签</p>

    <h3>这是一个相邻兄弟选择器</h3>

    <p>这是另外一个p标签</p>

  

    <h3 id="element">这是一个伪类选择器</h3>

</body>

</html>
```

# 块、行捏、行内块元素
## 块元素（block）:
- 块级元素通常会从新行开始，并占据整行的宽度。
- 可以包含其他块级元素和行内元素。
## 行内元素（inline）:
- 行内元素通常在同一行内呈现，不会独占一行。
- 它们只占据其内容所需的宽度，而不是整行的宽度。
- 行内元素不能包含块级元素，但可以包含其他行内元素。
## 行内块元素（Inline-block）:
- 水平方向上排列，但可以设置宽度、高度、内外边距等块级元素的属性。
- 行内块元素可以包含其他行内元素或块级元素。
```html
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>CSS 常用属性</title>

    <style>

        .block{

            display: block;

            width: 100px;

            height: 100px;

            background-color: red;

        }

        .inline{

            display: inline;

            width: 100px;

            /* 行内元素不支持宽 */

            height: 100px;

            background-color: blue;

        }

        .inline-block{

            display: inline-block;

            width: 100px;

            height: 100px;

            background-color: green;

        }

        .div-inline{

            display: inline;

            width: 100px;

            height: 100px;

            background-color: yellow;

        }

        .span-inline-block{

            display: inline-block;

            width: 100px;

            height: 100px;

            background-color: pink;

        }

        #element:hover{

            color: orange;

        }

    </style>

</head>

<body>

    <h1 style="font: bolder 50px 'kaiti';">这是一个font符合属性示例</h1>

    <p style="line-height: 100px;">____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________</p>

    <div class="block">这是一个块级元素</div>

    <div class="inline">这是一个行内元素</div>

    <img src="./img.png" alt="" class="inline-block">

    <h2>display</h2>

    <div class="div-inline" id="element">这是一个转换或行内元素的div 标签</div>

    <span class="span-inline-block">这是一个转换成行内元素的span标签</span>

  

</body>

</html>

```

# 盒子模型
## 盒子模型相关属性

|     属性名      | 说明                                       |
| :----------: | ---------------------------------------- |
| 内容（Content）  | 盒子包含的实际内容，比如文本、图片等。                      |
| 内边距（Padding） | 围绕在内容的内部，是内容与边框之间的空间。可以使用padding属性来设置。   |
|  边框（Border）  | 围绕在内边距的外部，是盒子的边界。可以使用border属性来设置。        |
| 外边距（Margin）  | 围绕在边框的外部，是盒子与其他元素之间的空间。可以使用~margin属性来设置。 |
```html
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>CSS 盒子模型</title>

    <style>

        .demo{

            background-color: aqua;

            display: inline-block;

            /* solid是实线，还可以是dotted虚线，double双线，groove凹槽线，ridge凸槽线，inset内嵌线，outset外凸线 */

            border: 5px solid red;

            padding: 50px;

            margin: 30px;

        }

        .border-demo{

            background-color: yellow;

            /* border-style: solid; */

            border-style: solid dashed dotted double;

            border-color: blueviolet;

            width: 100px;

            height: 100px;

            /* padding: 20px;

            margin: 20px; */

            /* border-width: 10px; */

            border-width: 10px 2px 30px 20px;

        }

  

    </style>

</head>

<body>

    <div class="demo">hello</div>

    <div class="border-demo">这是一边框示例</div>

</body>

</html>
```

# 浮动
## 传统网页布局方式
在学习浮动之前，先了解传统的网页布局方式
网页布局方式有以下五种：
- 标准流（普通流、文档流）：网页按照元素的书写顺序依次排列
- 浮动
- 定位Flexbox和Grid（自适应布局）
标准流是由块级元素和行内元素按照默认规定的方式来排列，块级就是占一行，行内元素一行放好多个元素。

## 浮动
元素脱离文档流，根据开发者的意愿漂浮到网页的任意方向。
浮动属性用于创建浮动框，将其移动到一边，直到左边缘或右边缘触及包含块或另一个浮动框的边缘，这样即可使得元素进行浮动。
语法：
```html
选择器 {float: left/right/none;}
```

注意：浮动是相对于父元素浮动，只会在父元素的内部移动

### 浮点三大特性

学习浮动要先了解浮动的三大特性：
- 脱标：脱离标准流。
- 一行显示，顶部对齐
- 具备行内块元素特性
```
<!DOCTYPE html>

<html lang="en">

<head>html

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>浮动</title>

    <style>

        .father{

            /* width: 500px;

            height: 300px; */

            background-color: aqua;

            border: 3px;

            /* overflow: hidden; */

        }

        .father::after{

            content: '';

            display: block;

            clear: both;

        }

        .leftson{

            width: 200px;

            height: 200px;

            background-color: red;

            float: left;

        }

        .rightson{

            width: 200px;

            height: 200px;

            background-color: blue;

            float: right;

        }

    </style>

</head>

<body>

    <div class="father">

        <div class="leftson">左浮动</div>

        <div class="rightson">右浮动</div>

        <p>aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</p>

    </div>

</body>

</html>
```

## 定位
定位布局可以精准定位，但缺乏灵活性
定位方式：
- 相对定位：相对于元素在文档流中的正常位置进行定位。
- 绝对定位：相对于其最近的已定位祖先元素进行定位，不占据文档流。
- 固定定位：相对于浏览器窗口进行定位。不占据文档流，固定在屏幕上的位置，不随滚动而移动。
```html
<!DOCTYPE html>

<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <title>定位</title>

    <style>

        .box1{

            /* width: 500px; */

            height: 500px;

            background-color: aqua;

            position: relative;

        }

        .box-normal{

            width: 100px;

            height: 100px;

            background-color: red;

        }

        .box-relative{

            width: 100px;

            height: 100px;

            background-color: blue;

            position: relative;

            top: 0px;

            left: 50px;

        }

  

        .box2{

            /* width: 500px; */

            height: 500px;

            background-color: yellow;

            position: relative;

        }

        .box-absolute{

            width: 100px;

            height: 100px;

            background-color: green;

            position: absolute;

            top: 111px;

            left: 50px;

        }

  

        .box-fix{

            width: 100px;

            height: 100px;

            background-color: pink;

            position: fixed;

            top: 0px;

            right: 0px;

        }

    </style>

</head>

<body>

    <h1>相对定位</h1>

    <div class="box1">

        <div class="box-normal"></div>

        <div class="box-relative"></div>

        <div class="box-normal"></div>

    </div>

    <h1>绝对定位</h1>

    <div class="box2">

        <div class="box-normal"></div>

        <div class="box-absolute"></div>

        <div class="box-normal"></div>

    </div>

    <h1>固定定位</h1>

    <div class="box-fix"></div>

</body>

</html>
```