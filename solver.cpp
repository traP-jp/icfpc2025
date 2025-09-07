#include <bits/stdc++.h>
using namespace std;

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>

// nlohmann/jsonを使いやすくするために名前空間エイリアスを設定
using json = nlohmann::json;

using ll = long long;
using ull = unsigned long long;
using vecint = std::vector<int>;
using vecll = std::vector<long long>;
using vecstr = std::vector<string>;
using vecbool = std::vector<bool>;
using vecdou = std::vector<double>;
using vecpl = std::vector<pair<ll,ll>>;
using vec2d = std::vector<vecll>;

const string BASE_URL = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com";
const string TEAM_ID = "ここにチームIDを入力";
const int SIZE = 24;
const string problemName = "beth";
vector<vector<int>> graph(SIZE, vector<int>(6, -1));
vector<int> labels(SIZE,-1);
vector<vector<int>> color_vertex(4);
int vertex_count = 0;

struct connection {
    int from_room;
    int from_door;
    int to_room;
    int to_door;
    connection(int fr, int fd, int tr, int td)
        : from_room(fr), from_door(fd), to_room(tr), to_door(td) {}
};
vector<connection> connections;

void select() {
    std::cout << "--- Selecting problem: " << problemName << " ---" << std::endl;
    
    // リクエストボディを作成
    json request_body = {
        {"id", TEAM_ID},
        {"problemName", problemName}
    };

    // APIリクエストを送信
    cpr::Response r = cpr::Post(
        cpr::Url{BASE_URL + "/select"},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{request_body.dump()}
    );

    if (r.status_code == 200) {
        std::cout << "Problem selected successfully." << std::endl;
        std::cout << "Response: " << r.text << std::endl;
    } else {
        std::cerr << "Failed to select problem. Status code: " << r.status_code << std::endl;
        std::cerr << "Error: " << r.text << std::endl;
    }
}
vector<std::vector<int>> explore(const std::vector<std::string>& plans) {
    std::cout << "\n--- Exploring the library... ---" << std::endl;

    // リクエストボディを作成
    json request_body = {
        {"id", TEAM_ID},
        {"plans", plans}
    };

    // APIリクエストを送信
    cpr::Response r = cpr::Post(
        cpr::Url{BASE_URL + "/explore"},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{request_body.dump()}
    );

    if (r.status_code == 200) {
        json response_body = json::parse(r.text);
        std::cout << "Exploration successful." << std::endl;
        std::cout << "Response: " << response_body.dump(2) << std::endl;
        
        // JSONの "results" フィールドを vector<vector<int>> に変換して返す
        return response_body["results"].get<std::vector<std::vector<int>>>();
    } else {
        std::cerr << "Failed to explore. Status code: " << r.status_code << std::endl;
        std::cerr << "Error: " << r.text << std::endl;
        return {}; // エラー時は空のベクタを返す
    }
}
void guess() {
    std::cout << "\n--- Submitting guess based on discovered map... ---" << std::endl;

    // 1. APIの "map" オブジェクトを構築
    json map_data;

    // 1-1. "rooms" フィールドを設定
    // グローバル変数 `labels` をそのまま使用します。
    map_data["rooms"] = labels;

    // 1-2. "startingRoom" フィールドを設定 (プロンプトの指示通り0に固定)
    map_data["startingRoom"] = 0;

    // 1-3. "connections" フィールドを設定
    // グローバル変数 `connections` からAPIの形式に変換します。
    json connection_list = json::array();
    for (const auto& conn : connections) {
        connection_list.push_back({
            {"from", {{"room", conn.from_room}, {"door", conn.from_door}}},
            {"to", {{"room", conn.to_room}, {"door", conn.to_door}}}
        });
    }
    map_data["connections"] = connection_list;


    // 2. リクエスト全体のボディを作成
    json request_body = {
        {"id", TEAM_ID}, // TEAM_IDはグローバル変数として定義されている想定
        {"map", map_data}
    };

    // デバッグ用に送信するJSONを表示
    std::cout << "Sending JSON:\n" << request_body.dump(2) << std::endl;

    // 3. APIリクエストを送信
    cpr::Response r = cpr::Post(
        cpr::Url{BASE_URL + "/guess"},
        cpr::Header{{"Content-Type", "application/json"}},
        cpr::Body{request_body.dump()}
    );

    // 4. レスポンスを処理
    if (r.status_code == 200) {
        json response_body = json::parse(r.text);
        std::cout << "Guess submitted successfully." << std::endl;
        if (response_body["correct"]) {
            std::cout << "Result: CORRECT! 🎉" << std::endl;
        } else {
            std::cout << "Result: INCORRECT. 😞" << std::endl;
        }
    } else {
        std::cerr << "Failed to submit guess. Status code: " << r.status_code << std::endl;
        std::cerr << "Error: " << r.text << std::endl;
    }
}

string UtoV(int u,int v){
    vector<int> dist(SIZE, -1);
    queue<int> q;
    q.push(u);
    dist[u] = 0;
    while(!q.empty()){
        int cur = q.front(); q.pop();
        for(int to: graph[cur]){
            if(to == -1) continue;
            if(dist[to] != -1) continue;
            dist[to] = dist[cur] + 1;
            q.push(to);
        }
    }
    string res = "";
    while(v != u){
        for(int to: graph[v]){
            if(to == -1) continue;
            if(dist[to] + 1 == dist[v]){
                for(int i=0;i<6;i++){
                    if(graph[to][i] == v){
                        res += (char)('0' + i);
                        break;
                    }
                }
                v = to;
                break;
            }
        }
    }
    reverse(res.begin(), res.end());
    return res;
}

void identify_edges(int vertex) {
    for (int i=0;i<6;i++){
        if(graph[vertex][i] != -1) continue;
        string initial_path = UtoV(0, vertex) + (char)('0' + i);
        vector<vector<int>> tmp = explore({initial_path});
        int c = tmp[0].back();
        int to_idx=-1;
        for(int u: color_vertex[c]){
            string Q = UtoV(0, u);
            Q += '[';
            Q += (char)('0' + (c+1)%4);
            Q += ']';
            Q += UtoV(u, vertex);
            Q += (char)('0' + i);
            vector<vector<int>> res = explore({Q});
            if(res[0].back() != c){
                to_idx = u;
                break;
            }
        }
        if(to_idx == -1){
            to_idx = vertex_count;
            labels[to_idx] = c;
            color_vertex[c].emplace_back(to_idx);
            vertex_count++;
        }
        for(int j=0;j<6;j++){
            if(graph[to_idx][j] != -1) continue;
            string Q1 = UtoV(0,vertex);
            Q1 += (char)('0' + i);
            Q1 += (char)('0' + j);
            string Q2 = UtoV(0,vertex);
            Q2 += '[';
            Q2 += (char)('0' + (labels[vertex]+1)%4);
            Q2 += ']';
            Q2 += (char)('0' + i);
            Q2 += (char)('0' + j);
            vector<vector<int>> res = explore({Q1, Q2});
            if(res[0].back() == labels[vertex] && res[1].back() != labels[vertex]){
                graph[vertex][i] = to_idx;
                graph[to_idx][j] = vertex;
                connections.emplace_back(vertex, i, to_idx, j);
                break;
            }
        }
    }
}

int main() {
    select();
    vector<vector<int>> tmp = explore({""});
    labels[0] = tmp[0][0];
    color_vertex[labels[0]].emplace_back(0);
    vertex_count++;
    for(int i=0; i<SIZE; i++){
        identify_edges(i);
    }
    guess();
}
