import math


def factorial(x):
    """
    计算阶乘函数，可以处理整数和半整数
    """
    if x < 0:
        return 0
    # 使用Gamma函数计算阶乘，适用于整数和半整数
    return math.gamma(x + 1)


def is_half_integer(x):
    """检查是否为整数或半整数"""
    return abs(2 * x - round(2 * x)) < 1e-10


def triangle_inequality(j1, j2, j):
    """检查三角不等式 |j1-j2| <= j <= j1+j2"""
    return abs(j1 - j2) - 1e-10 <= j <= j1 + j2 + 1e-10


def clebsch_gordan(j1, j2, j, m1, m2, m):
    """
    计算Clebsch-Gordan系数 ⟨j1 j2 m1 m2|j1 j2 j m⟩
    """
    # 检查基本选择定则
    if not (is_half_integer(j1) and is_half_integer(j2) and is_half_integer(j) and
            is_half_integer(m1) and is_half_integer(m2) and is_half_integer(m)):
        return 0.0

    if abs(m1 + m2 - m) > 1e-10:
        return 0.0

    if abs(m1) > j1 + 1e-10 or abs(m2) > j2 + 1e-10 or abs(m) > j + 1e-10:
        return 0.0

    if not triangle_inequality(j1, j2, j):
        return 0.0

    # 使用Racah公式计算CG系数
    try:
        term1 = (2 * j + 1) * factorial(j1 + j2 - j) * factorial(j1 - j2 + j) * factorial(-j1 + j2 + j)
        term1 /= factorial(j1 + j2 + j + 1)
        term1 = math.sqrt(term1)

        term2 = factorial(j1 + m1) * factorial(j1 - m1) * factorial(j2 + m2) * factorial(j2 - m2)
        term2 *= factorial(j + m) * factorial(j - m)
        term2 = math.sqrt(term2)

        # 求和范围
        k_min = max(0.0, j2 - j - m1, j1 - j + m2)
        k_max = min(j1 + j2 - j, j1 - m1, j2 + m2)

        total = 0.0
        for k in range(int(k_min), int(k_max) + 1):
            denom = (factorial(k) * factorial(j1 + j2 - j - k) *
                     factorial(j1 - m1 - k) * factorial(j2 + m2 - k) *
                     factorial(j - j2 + m1 + k) * factorial(j - j1 - m2 + k))
            if denom == 0:
                continue
            term = (-1.0) ** k / denom
            total += term

        result = term1 * term2 * total
        return result
    except:
        return 0.0


def three_j_symbol(j1, j2, j, m1, m2, m):
    """
    计算3j符号
    """
    # 3j符号要求 m1 + m2 + m = 0
    if abs(m1 + m2 + m) > 1e-10:
        return 0.0

    cg = clebsch_gordan(j1, j2, j, m1, m2, -m)
    if abs(cg) < 1e-10:
        return 0.0

    phase = (-1.0) ** (j1 - j2 - m)
    return phase * cg / math.sqrt(2 * j + 1)


def run_test_cases():
    """运行测试样例"""
    print("Clebsch-Gordan系数和3j符号计算器")
    print("=" * 50)

    # 测试样例
    test_cases = [
        # (j1, j2, j, m1, m2, m, 描述)
        (1, 1, 2, 1, 1, 2, "最大投影角动量"),
        (1, 1, 2, 1, 0, 1, "中间投影角动量"),
        (1, 1, 0, 1, -1, 0, "角动量为0的情况"),
        (0.5, 0.5, 1, 0.5, 0.5, 1, "自旋1/2系统"),
        (1, 0.5, 1.5, 1, 0.5, 1.5, "混合整数半整数"),
        (1, 1, 1, 1, 0, 1, "不满足选择定则的例子"),
        # 新增测试案例，满足3j符号条件 m1+m2+m=0
        (1, 1, 2, 1, 1, -2, "3j符号测试1"),
        (1, 1, 1, 1, -1, 0, "3j符号测试2"),
        (0.5, 0.5, 1, 0.5, -0.5, 0, "3j符号测试3")
    ]

    print("测试样例及结果:")
    print("-" * 50)

    for j1, j2, j, m1, m2, m, desc in test_cases:
        cg = clebsch_gordan(j1, j2, j, m1, m2, m)
        three_j = three_j_symbol(j1, j2, j, m1, m2, m)

        print(f"案例: {desc}")
        print(f"输入: j1={j1}, j2={j2}, j={j}, m1={m1}, m2={m2}, m={m}")
        print(f"CG系数: {cg:.10f}")
        print(f"3j符号: {three_j:.10f}")
        print("-" * 40)


def interactive_input():
    """交互式输入函数"""
    print("\n" + "=" * 60)
    print("现在进入交互式输入模式")
    print("请输入量子数 (可以是整数或半整数，如 1, 0.5, 1.5 等)")
    print("按 Ctrl+C 退出程序")
    print("=" * 60)

    while True:
        try:
            print("\n请输入量子数:")
            j1 = float(input("j1 = "))
            j2 = float(input("j2 = "))
            j = float(input("j = "))
            m1 = float(input("m1 = "))
            m2 = float(input("m2 = "))
            m = float(input("m = "))

            # 计算CG系数和3j符号
            cg = clebsch_gordan(j1, j2, j, m1, m2, m)
            three_j = three_j_symbol(j1, j2, j, m1, m2, m)

            print("\n计算结果:")
            print(f"Clebsch-Gordan系数: {cg:.10f}")
            print(f"3j符号: {three_j:.10f}")

        except KeyboardInterrupt:
            print("\n\n程序已退出。")
            break
        except ValueError:
            print("错误: 请输入有效的数字!")
        except Exception as e:
            print(f"计算错误: {e}")


def main():
    """主函数"""
    # 首先运行测试样例
    run_test_cases()

    # 然后进入交互式输入模式
    interactive_input()


if __name__ == "__main__":
    main()