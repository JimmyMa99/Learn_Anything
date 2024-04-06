// alert('tabel.js loaded')
// 新增数据函数
function addrow() {
    // 获取表格
    var table = document.getElementById('table');
    // 获取表格行数
    var rows = table.rows.length;
    // 创建新行
    var newRow = table.insertRow(rows);
    // 创建新单元格
    var newCell1 = newRow.insertCell(0);
    var newCell2 = newRow.insertCell(1);
    var newCell3 = newRow.insertCell(2);
    // 添加内容
    newCell1.innerHTML = '未命名';
    newCell2.innerHTML = '无联系方式';
    newCell3.innerHTML = '<button onclick="delrow(this)">删除</button> <button onclick="editrow(this)"">修改</button>';
}
// 删除数据函数
function delrow(button) {
    // // 获取表格
    var table = document.getElementById('table');
    // // 获取表格行数
    // var rows = table.rows.length;
    // // 删除最后一行
    // table.deleteRow(button.count);
    var row=button.parentNode.parentNode;
    // console.log(row)
    row.parentNode.removeChild(row);
}

// 修改数据函数
function editrow(button) {
    // 获取表格
    var table = document.getElementById('table');
    // 获取表格行数
    // var rows = table.rows.length;
    // 获取当前行
    var row = button.parentNode.parentNode;
    // 获取当前行的单元格
    var cells = row.cells;
    // 修改内容
    // cells[0].innerHTML = '<input type="text" value="' + cells[0].innerHTML + '">';
    // cells[1].innerHTML = '<input type="text" value="' + cells[1].innerHTML + '">';
    // cells[0].innerHTML = prompt("请输入姓名",cells[0].innerHTML);
    // cells[1].innerHTML = prompt("请输入联系方式",cells[1].innerHTML);
    // cells[2].innerHTML = '<button onclick="saverow(this)">保存</button> <button onclick="cancelrow(this)">取消</button>';
    var name = cells[0];
    var contact = cells[1];

    var inputName = prompt("请输入姓名", name.innerHTML);
    var inputContact = prompt("请输入联系方式", contact.innerHTML);

    name.innerHTML = inputName;
    contact.innerHTML = inputContact;
}

// 保存数据函数
function saverow(button) {
    // 获取表格
    var table = document.getElementById('table');
    // 获取当前行
    var row = button.parentNode.parentNode;
    // 获取当前行的单元格
    var cells = row.cells;
    // 保存内容
    cells[0].innerHTML = cells[0].getElementsByTagName('input')[0].value;
    cells[1].innerHTML = cells[1].getElementsByTagName('input')[0].value;
    cells[2].innerHTML = '<button onclick="delrow(this)">删除</button> <button onclick="editrow(this)">修改</button>';
}