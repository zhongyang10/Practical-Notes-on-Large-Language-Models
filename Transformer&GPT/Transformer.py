import numpy as np
import torch
import torch.nn as nn
import yaml


class ConfigReader:
    def __init__(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                if not isinstance(config, dict):
                    raise ValueError("配置文件解析结果不是字典类型，请检查 YAML 文件格式。")
                self.DimQ = config.get('DimQ')
                self.DimK = config.get('DimK')
                self.DimV = config.get('DimV')
                self.DimeEmb = config.get('DimeEmb')
                self.NumHeads = config.get('NumHeads')
                self.BatchSize = config.get('BatchSize')
                self.Channels = config.get('Channels')
                self.NumLayers = config.get('NumLayers')
        except FileNotFoundError:
            print(f"错误：未找到配置文件 {config_path}。")
        except yaml.YAMLError as e:
            print(f"错误：无法解析 YAML 配置文件，错误信息：{e}。")
        except ValueError as e:
            print(e)


class ScalDotProductAttention(nn.Module):
    def __init__(self):
        super(ScalDotProductAttention,self).__init__()
    def forward(self,Q,K,V,mask):
        #计算注意力分数并进行缩放
        pass

        

