def read_log():
    """
    读取日志文件,进行数据重组,写入mysql
    :return:
    """
    file = "04022022.log"
    with open(file) as f:
        """使用while循环每次只读取一行,读到最后一行的时候结束"""
        while True:
            lines = f.readline()
            if not lines:
                break
            line = lines.split(",")
            data.append((line[0], line[1].strip(), int(str(line[2]).strip()), line[3].strip()))
            return data


if __name__ == '__main__':
    data = []
    print(read_log())
