# from fnnconfig.utils.str2cls import dict2cls
import importlib


class Config():
    def __init__(self, config: str):

        config = importlib.import_module(config)
        print(config.model)
        # print(config.train_dataloader)
        # print(config._base_)
        # print(config.__dict__)
        # print(config.train_dataloader)



        # with open(config, 'r') as cf:
        #     cf_content = cf.read()

        # evaluated_content = eval(compile(cf_content, config, 'exec'))

        # # with open(filename, 'r') as file:
        # #     file_content = file.read()

        # # evaluated_content = eval(compile(file_content, filename, 'exec'))

        # print(evaluated_content)


    def merge(self, d):

        pass

if __name__ == "__main__":



    # c = Config('/home/peter/workspace/scratch/fnn/fnnconfig/src/fnnconfig/yolo/yolox_base_b16.py')
    c = Config('fnnconfig.yolo.yolox_base_b16')
    # # print(c.__dict__)
    # # print(c.train_dataloader)

    # config = importlib.import_module('fnnconfig.yolo.yolox_base_b16')


    # import importlib  
    
    # def get_module_locals(module_name):  
    #     # 动态导入模块  
    #     module = importlib.import_module(module_name)  
        
    #     # 定义一个函数来执行模块的代码
    #     def execute_module():  
    #         exec(module.__code__.co_source, globals(), locals())  
    #         return locals()  
        
    #     # 返回模块的局部变量字典
    #     return execute_module()  
    
    # # 使用示例  
    # config = importlib.import_module('fnnconfig.yolo.yolox_base_b16')  
    # config_locals = get_module_locals('fnnconfig.yolo.yolox_base_b16')  
    # print(config_locals)
