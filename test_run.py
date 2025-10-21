import subprocess
import json
from typing import List, Dict, Any
def evaluate_performance(program_name: str, run_args: str, run_count: int = 1) -> dict:
    # 构建命令列表（拆分参数字符串为列表元素）
    command = [program_name] + run_args.split()
    
    # 存储每次运行的性能数据
    tok_per_s_list: List[float] = []
    ftl_per_ms_list: List[float] = []
    memory_usage_list: List[float] = []
    power_consumption_list: List[float] = []
    
    def _run_single() -> Dict[str, float]:
        """单次运行程序并返回性能数据"""
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"第{len(tok_per_s_list)+1}次运行失败: {e.stderr}") from e
        except FileNotFoundError:
            raise FileNotFoundError(f"未找到程序: {program_name}")
        
        output_lines = result.stderr.splitlines()
        if len(output_lines) < 4:
            raise ValueError(f"第{len(tok_per_s_list)+1}次运行输出内容不足，无法提取性能数据")
        
        performance_lines = output_lines[-4:]
        try:
            return {
                'tok_per_s': float(performance_lines[0].split(': ')[1]),
                'ftl_per_ms': float(performance_lines[1].split(': ')[1]),
                'memory_usage': float(performance_lines[2].split(': ')[1]),
                'power_consumption': float(performance_lines[3].split(': ')[1])
            }
        except (IndexError, ValueError) as e:
            raise ValueError(f"第{len(tok_per_s_list)+1}次运行性能数据格式错误: {e}") from e
    
    # 执行多次运行
    for _ in range(run_count):
        single_result = _run_single()
        tok_per_s_list.append(single_result['tok_per_s'])
        ftl_per_ms_list.append(single_result['ftl_per_ms'])
        memory_usage_list.append(single_result['memory_usage'])
        power_consumption_list.append(single_result['power_consumption'])
    
    # 计算平均值
    def _average(data_list: List[float]) -> float:
        return sum(data_list) / len(data_list)
    
    return {
        'program_name': program_name,
        'run_args': run_args,
        'run_count': run_count,  # 新增：记录实际运行次数
        'tok_per_s_avg': _average(tok_per_s_list),
        'ftl_per_ms_avg': _average(ftl_per_ms_list),
        'memory_usage_avg': _average(memory_usage_list),
        'power_consumption_avg': _average(power_consumption_list)
    }
def save_performance_results(results: List[Dict[str, Any]], file_path: str) -> None:
    # 验证输入是否为字典列表
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise TypeError("输入必须是字典组成的列表")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # indent=4 用于格式化输出，增强可读性
            json.dump(results, f, ensure_ascii=False, indent=4)
    except IOError as e:
        raise IOError(f"保存文件失败: {str(e)}") from e
    except TypeError as e:
        raise TypeError(f"结果包含不支持的JSON数据类型: {str(e)}") from e


def load_performance_results(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 验证读取结果是否为字典列表
        if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
            raise ValueError("文件内容格式错误，应为字典组成的列表")
        
        return results
    except FileNotFoundError:
        raise FileNotFoundError(f"文件不存在: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"文件格式错误，不是有效的JSON: {str(e)}", e.doc, e.pos) from e
    except IOError as e:
        raise IOError(f"读取文件失败: {str(e)}") from e

def run_testcases(cases: List[Dict[str, Any]], output_path: str) -> None:
    ans=list()
    cnt=0
    for case in cases:
        cnt+=1
        print(f"Running case {cnt}...")
        curAns = evaluate_performance(case['prg'], case['args'], 10)
        ans.append(curAns)
    save_performance_results(ans, output_path)

testCases = [{'prg':'./run_int8.out','args':'./modelq_110M.bin' },
         {'prg':'./run_int16.out','args':'./modelq16_110M.bin' },
         {'prg':'./run_fp32.out','args':'./model_110M.bin' },
         {'prg':'./run_no_paral.out','args':'./modelq_110M.bin' },
         {'prg':'./run_paral.out','args':'./modelq_110M.bin' },
         {'prg':'./run_avx.out','args':'./modelq_110M.bin' },
         {'prg':'./run_avx_plus.out','args':'./modelq_110M.bin' },
         {'prg':'./run_int8.out','args':'./modelq_15M.bin' },
         {'prg':'./run_int8.out','args':'./modelq_42M.bin' }
        ]

run_testcases(testCases, "output.json")
