def parse_obj_file():
    file_path = r'E:\ABB-Project\ABB_wrs_hu\suyixuan\textured_mesh.obj'
    vertex_count = 0
    face_count = 0
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # 去除行首尾的空白字符
                line = line.strip()
                # 检查顶点行
                if line.startswith('v '):
                    vertex_count += 1
                # 检查面行
                elif line.startswith('f '):
                    face_count += 1
        print(f"顶点数: {vertex_count}")
        print(f"面数: {face_count}")
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")
    except Exception as e:
        print(f"发生错误: {e}")


if __name__ == "__main__":
    parse_obj_file()
