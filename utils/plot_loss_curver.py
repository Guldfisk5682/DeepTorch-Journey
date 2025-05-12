import matplotlib.pyplot as plt
# 由AI完成
class LiveLossPlotter:
    """
    一个用于实时绘制训练过程中损失曲线的类。
    可以在每个epoch后更新图像，并支持绘制单条或多条曲线（例如训练损失和验证损失）。
    """

    def __init__(self, title="loss curver", xlabel="Epoch", ylabel="Loss", multiple_lines=None):
        """
        初始化绘图器。

        使用方法:
        单条曲线:

        多条曲线 (例如, 训练损失和验证损失):
        plotter = LiveLossPlotter(
            title="训练与验证损失",
            xlabel="Epoch",
            ylabel="Loss",
            multiple_lines={"train_loss": "b-o", "val_loss": "g--s"} # "b-o": 蓝色实线带圆点, "g--s": 绿色虚线带方块
        )
        multiple_lines 是一个字典，键是曲线的名称，值是matplotlib的线条样式字符串。

        Args:
            title (str): 图像的标题。
            xlabel (str): X轴的标签。
            ylabel (str): Y轴的标签。
            multiple_lines (dict, optional): 用于配置多条曲线。
                                            键为曲线名称 (str)，值为matplotlib线条样式 (str)。
                                            如果为 None，则默认绘制一条名为 "Loss" 的曲线。
        """
        plt.ion()  # 开启交互模式
        self.fig, self.ax = plt.subplots()

        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

        self.multiple_lines_config = multiple_lines
        self.epoch_data = {} # 存储所有曲线的epoch数据
        self.loss_data = {}  # 存储所有曲线的loss数据
        self.lines = {}      # 存储所有曲线的Line2D对象

        if self.multiple_lines_config:
            for name, style in self.multiple_lines_config.items():
                self.epoch_data[name] = []
                self.loss_data[name] = []
                self.lines[name], = self.ax.plot(self.epoch_data[name], self.loss_data[name], style, label=name)
            self.ax.legend() # 显示图例
        else:
            # 单条曲线模式
            default_line_name = "Loss"
            self.epoch_data[default_line_name] = []
            self.loss_data[default_line_name] = []
            self.lines[default_line_name], = self.ax.plot(self.epoch_data[default_line_name], self.loss_data[default_line_name], 'r-o', label=default_line_name) # 默认红色实线带圆点
            # 如果只有一条线，也可以选择不显示图例，或者显示
            # self.ax.legend()

        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, epoch, loss_values):
        """
        更新绘图数据并刷新图像。

        使用方法:
        在每个epoch结束时调用此方法。

        单条曲线:
        plotter.update(epoch_number, current_loss_value)
        例如: plotter.update(1, 0.75)

        多条曲线:
        loss_dict = {"train_loss": train_loss_value, "val_loss": val_loss_value}
        plotter.update(epoch_number, loss_dict)
        例如: plotter.update(1, {"train_loss": 0.75, "val_loss": 0.80})

        Args:
            epoch (int): 当前的epoch号。
            loss_values (float or dict):
                - 如果是单条曲线模式 (multiple_lines=None)，则为单个损失值 (float)。
                - 如果是多条曲线模式，则为一个字典，键为曲线名称 (str)，值为对应的损失值 (float)。
                  键必须与初始化时 multiple_lines 中的键一致。
        """
        if self.multiple_lines_config:
            if not isinstance(loss_values, dict):
                raise ValueError("对于多条曲线模式，loss_values 必须是一个字典。")
            for name, value in loss_values.items():
                if name in self.lines:
                    self.epoch_data[name].append(epoch)
                    self.loss_data[name].append(value)
                    self.lines[name].set_xdata(self.epoch_data[name])
                    self.lines[name].set_ydata(self.loss_data[name])
                else:
                    print(f"警告: 曲线 '{name}' 未在初始化时定义，将被忽略。")
        else:
            if isinstance(loss_values, dict):
                raise ValueError("对于单条曲线模式，loss_values 必须是一个浮点数。")
            default_line_name = "Loss" # 与__init__中一致
            self.epoch_data[default_line_name].append(epoch)
            self.loss_data[default_line_name].append(loss_values)
            self.lines[default_line_name].set_xdata(self.epoch_data[default_line_name])
            self.lines[default_line_name].set_ydata(self.loss_data[default_line_name])

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_plot(self, filepath="loss_curve.png"):
        """
        将当前图像保存到文件。

        使用方法:
        plotter.save_plot("my_training_loss.png")

        Args:
            filepath (str): 保存图像的文件路径和名称。
        """
        self.fig.savefig(filepath)
        print(f"图像已保存到: {filepath}")

    def finalize(self, keep_open=True):
        """
        训练完成后调用，关闭交互模式并选择是否保持窗口打开。

        使用方法:
        plotter.finalize() # 默认保持窗口打开
        plotter.finalize(keep_open=False) # 训练结束后自动关闭窗口
        """
        plt.ioff()
        if keep_open:
            plt.show() # 阻塞，直到用户手动关闭图像窗口