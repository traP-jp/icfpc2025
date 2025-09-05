import argparse
import random
import subprocess
import sys

def is_connected(connections):
    n = len(connections)
    visited = [False] * n
    stack = [0]
    visited[0] = True
    while stack:
        room = stack.pop()
        for to_room, _ in connections[room]:
            if not visited[to_room]:
                visited[to_room] = True
                stack.append(to_room)
    return all(visited)

def generate_labyrinth(n):
    """ランダムに迷宮を生成"""
    while True:
        labels = [random.randint(0, 3) for _ in range(n)]
        connections = [[None] * 6 for _ in range(n)]
        # 未接続のドアリストを作成
        door_list = [(room, door) for room in range(n) for door in range(6)]
        random.shuffle(door_list)
        # 2つずつペアにして繋ぐ
        while len(door_list) > 0:
            index1 = 0
            index2 = random.randint(0, len(door_list) - 1)
            room1, door1 = door_list[index1]
            room2, door2 = door_list[index2]
            connections[room1][door1] = (room2, door2)
            connections[room2][door2] = (room1, door1)
            door_list.pop(max(index1, index2))
            if index1 != index2:
                door_list.pop(min(index1, index2))
        
        if is_connected(connections):
            break

    return labels, connections

def simulate_explore(labels, connections, plan):
    """ルートプランを実行してラベル列を返す"""
    cur_room = 0
    result = [labels[cur_room]]
    for ch in plan:
        door = int(ch)
        nxt_room, _ = connections[cur_room][door]
        cur_room = nxt_room
        result.append(labels[cur_room])
    return result

def check_answer(labels, connections, answer_labels, answer_connections):
    n = len(labels)
    # 番号の対応が正しいか確認
    f = [-1] * n
    f[0] = 0
    stack = [0]
    while stack:
        v = stack.pop()
        for d in range(6):
            to, _ = answer_connections[v][d]
            if f[to] == -1:
                # toの番号が未定義なら、対応する部屋を探す
                f[to] = connections[f[v]][d][0]
                stack.append(to)
            elif f[to] != connections[f[v]][d][0]:
                return False
            
    for i in range(n):
        if f[i] == -1:
            return False
        for j in range(n):
            if i != j and f[i] == f[j]:
                return False
    # ラベルと接続が正しいか確認
    for i in range(n):
        if answer_labels[i] != labels[f[i]]:
            return False
    
    for i in range(n):
        for d in range(6):
            to, door = answer_connections[i][d]
            if (f[to], door) != connections[f[i]][d]:
                return False
    return True

def run_judge(size, program, manual=False):
    labels, connections = generate_labyrinth(size)
    limit = 18 * size

    query_count = 0     # 合計探索長
    explore_calls = 0   # /explore 呼び出し回数

    if manual:
        print(f"[judge] 部屋数: {size}")
        print(f"[judge] ラベル: {labels}")
        print(f"[judge] 接続: {connections}")
        while True:
            q = int(input("探索回数qを入力 (解答フェーズは0): "))
            if q == 0:
                print("[judge] --- 解答フェーズ ---")
                label_line = input("提出ラベル列 (スペース区切り): ").strip()
                answer_labels = list(map(int, label_line.split()))

                answer_connections = []
                print(f"部屋ごとに接続情報を入力 (各部屋 {size} 行, 各行 12個の整数: to,door x6)")
                for i in range(size):
                    parts = list(map(int, input(f"部屋{i}: ").split()))
                    i_connection = []
                    for d in range(6):
                        i_connection.append((parts[2*d], parts[2*d+1]))
                    answer_connections.append(i_connection)

                correct = check_answer(labels, connections, answer_labels, answer_connections)
                print("[judge] CORRECT?", correct)
                score = query_count + explore_calls
                print("[judge] SCORE =", score)
                break
            else:
                print(f"[judge] --- 探索フェーズ (q={q}) ---")
                plans = []
                total_len = 0
                for i in range(q):
                    plan = input(f"plan[{i}] (ドア番号列): ").strip()
                    total_len += len(plan)
                    plans.append(plan)

                if total_len > limit:
                    print(f"[judge] ERROR: total route length {total_len} exceeds limit {limit}", file=sys.stderr)
                    break

                query_count += q
                explore_calls += 1

                for plan in plans:
                    res = simulate_explore(labels, connections, plan)
                    print("[judge] 探索結果:", " ".join(map(str, res)))
    else:
        proc = subprocess.Popen(
            program,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # 部屋数を送信
        print(size, file=proc.stdin, flush=True)

        while True:
            line = proc.stdout.readline()
            if not line:
                break
            q = int(line.strip())
            if q == 0:
                # --- 解答フェーズ ---
                label_line = proc.stdout.readline().strip()
                answer_labels = list(map(int, label_line.split()))

                answer_connections = []
                for i in range(size):
                    parts = list(map(int, proc.stdout.readline().split()))
                    i_connection = []
                    for d in range(6):
                        i_connection.append((parts[2*d], parts[2*d+1]))
                    answer_connections.append(i_connection)
                
                # 正誤判定
                
                correct = check_answer(labels, connections, answer_labels, answer_connections)
                print("[judge] CORRECT?" , correct)
                score = query_count + explore_calls  # ペナルティ込み
                print("[judge] SCORE =", score)
                break
            else:
                # --- 探索フェーズ ---
                plans = []
                total_len = 0
                for _ in range(q):
                    plan = proc.stdout.readline().strip()
                    total_len += len(plan)
                    plans.append(plan)

                if total_len > limit:
                    print(f"[judge] ERROR: total route length {total_len} exceeds limit {limit}", file=sys.stderr)
                    proc.kill()
                    break

                query_count += q
                explore_calls += 1

                for plan in plans:
                    res = simulate_explore(labels, connections, plan)
                    print(" ".join(map(str, res)), file=proc.stdin, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("program", nargs="*", help="program to run")
    parser.add_argument("--manual", action="store_true", help="manual mode (人が手で操作)")
    args = parser.parse_args()
    run_judge(args.size, args.program, manual=args.manual)

if __name__ == "__main__":
    main()
