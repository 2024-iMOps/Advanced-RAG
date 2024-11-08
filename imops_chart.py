import numpy as np
import matplotlib.pyplot as plt

def evals_chart(evaluation_results, save_dir):
    # 점수 추출
    acc_scores = [evaluation_results[i]['evaluation']['accuracy'] for i in range(len(evaluation_results))]
    con_scores = [evaluation_results[i]['evaluation']['conciseness'] for i in range(len(evaluation_results))]
    
    data = [acc_scores, con_scores]
    colors = ['#4CAF50', '#2196F3']

    # 플롯 설정
    plt.figure(figsize=(10, 6))

    # 박스 플롯 생성
    boxprops = dict(linestyle='-', linewidth=2, color='black')
    medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
    box = plt.boxplot(data, positions=[1, 2], widths=0.6, patch_artist=True,
                      boxprops=boxprops, medianprops=medianprops)

    # 박스 플롯 색상 및 투명도 적용
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # 데이터 포인트 추가
    for i, y in enumerate(data):
        x = np.random.normal(1 + i, 0.04, size=len(y))
        plt.scatter(x, y, alpha=0.9, color='black', edgecolors='white', s=100)

    # 축 레이블 및 제목 설정
    plt.xticks([1, 2], ['accuracy_scores', 'conciseness_scores'])
    plt.xlabel('Scores')
    plt.ylabel('Score Value')
    plt.title('RAG Evaluation Scores')
    plt.ylim(0, 10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # 이미지 저장
    plt.savefig(f'{save_dir}/evaluation_chart.png', dpi=300, bbox_inches='tight')
    plt.show()