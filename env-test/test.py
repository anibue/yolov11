from ultralytics import YOLO
import os

# 选择使用方式：
# 直接使用预训练模型
def test_pretrained_model():
    """使用预训练模型进行检测"""
    print("=== 使用预训练模型检测 ===")
    
    # 检查当前目录下是否存在模型文件
    model_path = "./yolo11n.pt"
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，将自动下载...")
        # 如果本地没有，YOLO会自动下载
        model_path = "yolo11n.pt"
    
    model = YOLO(model_path)
    
    # 使用本地图像进行测试
    results = model("env-test/bus.jpg")
    results[0].show()
    print("预训练模型检测完成")

# 训练自定义模型
def train_custom_model():
    """训练自定义模型"""
    print("=== 开始训练自定义模型 ===")
    
    # 检查当前目录下是否存在模型文件
    model_path = "./yolo11n.pt"
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，将自动下载...")
        model_path = "yolo11n.pt"
    
    # 加载预训练模型作为起点
    model = YOLO(model_path)
    
    # 在COCO8数据集上进行训练（示例）
    train_results = model.train(
        data="coco8.yaml",  # 数据集配置文件路径
        epochs=10,  # 减少训练周期用于测试
        imgsz=640,  # 训练图像尺寸
        device="cpu",  # 运行设备
        verbose=True  # 显示训练过程
    )
    
    # 评估模型性能
    metrics = model.val()
    print("训练完成，模型已保存")
    
    # 使用本地图像进行测试
    results = model("env-test/bus.jpg")
    results[0].show()

# 加载已训练的模型
def load_trained_model():
    """加载之前训练好的模型"""
    model_path = "runs/detect/train/weights/best.pt"  # 训练后模型路径
    
    if os.path.exists(model_path):
        print("=== 使用已训练的模型 ===")
        model = YOLO(model_path)
        results = model("env-test/bus.jpg")
        results[0].show()
    else:
        print("未找到已训练的模型，请先训练或使用预训练模型")

def verify_model():
    """验证模型是否正确加载"""
    model_path = "./yolo11n.pt"
    
    print("=== 验证模型加载 ===")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"模型文件路径: {model_path}")
    print(f"模型文件存在: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"模型文件大小: {file_size:.2f} MB")
    
    try:
        model = YOLO(model_path)
        print("模型加载成功")
        print(f"模型类型: {type(model)}")
        return True
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False

if __name__ == "__main__":
    # 首先验证模型
    if verify_model():
        # 根据需要选择运行方式
        
        # 快速测试 - 直接使用预训练模型
        test_pretrained_model()
        
        # 如果需要训练自定义模型，取消下面的注释
        # train_custom_model()
        
        # 如果已经有训练好的模型，取消下面的注释
        # load_trained_model()
    else:
        print("请先确保模型文件存在或网络连接正常")

    # 验证模型加载
    verify_model()
