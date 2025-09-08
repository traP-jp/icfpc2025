#![allow(non_snake_case)]


const N:usize=12;
const TOUR:usize=N*6*3;

const ITER:usize=1e7 as usize;
const DEBUG:bool=false;


fn main(){
    let mut cnt=[0;4];
    for i in 0..N{
        cnt[i%4]+=1;
    }
    run(&cnt,"aleph");
}



fn run(cnt:&[usize;4],name:&str){
    for _ in 0..50{
        eprintln!("start");

        select_problem(name);
        let plan=make_plan();
        let obs=explore(&[plan])[0];

        eprintln!("explore-end");
        
        if let Some(ans)=solve(&cnt,&plan,&obs){
            let f=ans.guess();
            if f{
                eprintln!("correct");
                break;
            } else{
                eprintln!("wrong");
            }
        } else{
            eprintln!("giveup");
        }
        eprintln!();
    }
}



fn simulate(cnt:&[usize;4],ty:usize){
    let mut correct=0;
    let mut wrong=0;
    let mut giveup=0;
    let start=get_time();

    for t in 0..ty{
        let map=Map::gen(&cnt);
        let plan=make_plan();
        let obs=map.explore(&[plan])[0];

        let res=solve(&cnt,&plan,&obs);
        eprint!("{} / {} : ",t,ty);
        
        if let Some(ans)=res{
            if map.is_isomorphic(&ans){
                // if map.isok(&ans){
                //     eprintln!("correct");
                // } else{
                //     eprintln!("yavai");
                // }
                eprintln!("correct");
                correct+=1;
            } else{
                eprintln!("wrong");
                wrong+=1;
            }
        } else{
            eprintln!("giveup");
            giveup+=1;
        }
    }
    let end=get_time();

    eprintln!();
    eprintln!("correct : {:.2} ({correct} / {ty})",correct as f64/ty as f64);
    eprintln!("wrong  : {:.2} ({wrong} / {ty})",wrong as f64/ty as f64);
    eprintln!("giveup : {:.2} ({giveup} / {ty})",giveup as f64/ty as f64);
    eprintln!("time   : {:.2} sec",(end-start)/ty as f64);
}



fn make_plan()->[usize;TOUR]{
    let mut it=0..;
    let mut plan=[();TOUR].map(|_|it.next().unwrap()%6);
    rnd::shuffle(&mut plan);
    plan
}



fn solve(cnt:&[usize;4],plan:&[usize;TOUR],obs:&[usize;TOUR+1])->Option<Map>{
    // obsがiの頂点idの候補はcand[i].0..cand[i].1
    let mut cand=[(0,cnt[0]);4];
    for i in 1..4{
        cand[i]=(cand[i-1].1,cand[i-1].1+cnt[i]);
    }

    let gen=|obs:usize|{
        rnd::range(cand[obs].0,cand[obs].1)
    };
    let gen_skip=|obs:usize,skip:usize|{
        rnd::range_skip(cand[obs].0,cand[obs].1,skip)
    };

    let mut tour=vec![!0;obs.len()];
    for i in 0..obs.len(){
        tour[i]=gen(obs[i]);
    }

    // 最大値を除く値の総和
    let get=|a:&[i64]|->(usize,i64){
        let mut sum=0;
        let mut arg=!0;
        let mut max=0;
        for i in 0..N{
            sum+=a[i];
            if max<a[i]{
                max=a[i];
                arg=i;
            }
        }
        (arg,sum-max)
    };
    
    let mut cnt=[[[0;N];6];N];
    for i in 0..TOUR{
        cnt[tour[i]][plan[i]][tour[i+1]]+=1;
    }

    let mut es=[[0i64;N];N];
    let mut score=0;
    for i in 0..N{
        for j in 0..6{
            let (arg,add)=get(&cnt[i][j]);
            score+=add;
            if arg!=!0{
                es[i][arg]+=1;
            }
        }
    }
    
    let mut deg=[0;N];
    for v in 0..N{
        deg[v]=(0..N).map(|u|es[u][v].max(es[v][u])).sum::<i64>();
        score+=(deg[v]-6).max(0);
    }
    
    let log=logs();
    let mut valid=0;
    let T0=0.3; // todo
    let T1=0.;

    let mut hi=vec![];
    
    for it in 0..ITER{
        let time=it as f64/ITER as f64;
        // let heat=T0*(T1/T0).powf(time);
        let heat=T0+(T1-T0)*time;
        let add=-heat*log[it&65535]; // 最小化

        let i=rnd::get(TOUR+1);
        let cv=tour[i];

        // todo : 部分的に改善する近傍に限定してる、これ大丈夫なのか怪しい
        let nv={
            let mut r=(0,0);
            let o=obs[i];
            let mut res=gen_skip(o,cv);
            
            if i==0{
                let nv=tour[i+1];
                let nd=plan[i];
                
                for v in cand[o].0..cand[o].1{
                    if v!=cv && cnt[v][nd][nv]>0{
                        let new=(1,rnd::next());
                        if r<new{
                            r=new;
                            res=v;
                        }
                    }
                }
            } else if i+1>=tour.len(){
                let pv=tour[i-1];
                let pd=plan[i-1];
                
                for v in cand[o].0..cand[o].1{
                    if v!=cv && cnt[pv][pd][v]>0{
                        let new=(1,rnd::next());
                        if r<new{
                            r=new;
                            res=v;
                        }
                    }
                }
            } else{
                let pv=tour[i-1];
                let pd=plan[i-1];
                let nv=tour[i+1];
                let nd=plan[i];
                
                for v in cand[o].0..cand[o].1{
                    let f=(cnt[pv][pd][v]>0) as u8+(cnt[v][nd][nv]>0) as u8;
                    if v!=cv && f>0{
                        let new=(f,rnd::next());
                        if r<new{
                            r=new;
                            res=v;
                        }
                    }
                }
            }

            res
        };

        macro_rules! add_es{
            ($u:expr,$v:expr,$add:expr)=>{{
                let old=es[$u][$v].max(es[$v][$u]);
                hi.push(($u,$v,$add));
                es[$u][$v]+=$add;
                let new=es[$u][$v].max(es[$v][$u]);
                let add_deg=new-old;
                let mut diff=-(deg[$u]-6).max(0);
                deg[$u]+=add_deg;
                diff+=(deg[$u]-6).max(0);
                if $u!=$v{
                    diff-=(deg[$v]-6).max(0);
                    deg[$v]+=add_deg;
                    diff+=(deg[$v]-6).max(0);
                }
                diff
            }}
        }
        
        let old_deg=deg;
        let mut new=score;
        hi.clear();

        if i!=0{
            let prev=tour[i-1];
            let pd=plan[i-1];

            let (arg,add)=get(&cnt[prev][pd]);
            new-=add;
            new+=add_es!(prev,arg,-1);
            
            cnt[prev][pd][cv]-=1;
            cnt[prev][pd][nv]+=1;

            let (arg,add)=get(&cnt[prev][pd]);
            new+=add;
            new+=add_es!(prev,arg,1);
        }
        
        if i+1<tour.len(){
            let next=tour[i+1];
            let nd=plan[i];

            let (argc,addc)=get(&cnt[cv][nd]);
            let (argn,addn)=get(&cnt[nv][nd]);
            new-=addc+addn;
            new+=add_es!(cv,argc,-1);
            if argn!=!0{
                new+=add_es!(nv,argn,-1);
            }

            cnt[cv][nd][next]-=1;
            cnt[nv][nd][next]+=1;
            
            let (argc,addc)=get(&cnt[cv][nd]);
            let (argn,addn)=get(&cnt[nv][nd]);
            new+=addc+addn;
            if argc!=!0{
                new+=add_es!(cv,argc,1);
            }
            new+=add_es!(nv,argn,1);
        }
        
        if (new-score) as f64<=add{ // 最小化
            score=new;
            valid+=1;
            tour[i]=nv;
            if new==0{
                break;
            }
        } else{
            deg=old_deg;
            
            if i+1<tour.len(){
                let next=tour[i+1];
                let nd=plan[i];
                
                cnt[cv][nd][next]+=1;
                cnt[nv][nd][next]-=1;
            }

            if i!=0{
                let prev=tour[i-1];
                let pd=plan[i-1];
                
                cnt[prev][pd][cv]+=1;
                cnt[prev][pd][nv]-=1;
            }

            for &(a,b,c) in &hi{
                es[a][b]-=c;
            }
        }
    }

    {
        let mut verify_cnt=[[[0;N];6];N];
        for i in 0..TOUR{
            verify_cnt[tour[i]][plan[i]][tour[i+1]]+=1;
        }
        assert!(verify_cnt==cnt);

        let mut verify_es=[[0i64;N];N];
        let mut verify_score=0;
        for i in 0..N{
            for j in 0..6{
                let (arg,add)=get(&verify_cnt[i][j]);
                verify_score+=add;
                if arg!=!0{
                    verify_es[i][arg]+=1;
                }
            }
        }
        assert!(verify_es==es);
        
        let mut verify_deg=[0;N];
        for v in 0..N{
            verify_deg[v]=(0..N).map(|u|verify_es[u][v].max(verify_es[v][u])).sum::<i64>();
            verify_score+=(verify_deg[v]-6).max(0);
        }
        assert!(verify_deg==deg);
        assert!(verify_score==score);
    }

    if DEBUG{
        eprintln!("ratio = {:.3}",valid as f64/ITER as f64);
        eprintln!("score = {}",score);
    }

    if score!=0{
        return None;
    }

    let mut g=[[!0;6];N];
    for i in 0..N{
        for j in 0..6{
            if let Some(k)=(0..N).find(|&k|cnt[i][j][k]!=0){
                assert!((0..N).all(|l|l==k || cnt[i][j][l]==0));
                g[i][j]=k;
            }
        }
    }

    let find=|a:&[usize;6]|{
        (0..6).find(|&d|a[d]==!0).unwrap()
    };

    for u in 0..N{
        for v in 0..u{
            if es[u][v]!=es[v][u]{
                let (mut u,mut v)=(u,v);
                if es[u][v]>es[v][u]{
                    (u,v)=(v,u);
                }
                for _ in 0..es[v][u]-es[u][v]{
                    let d=find(&g[u]);
                    g[u][d]=v;
                }
            }
        }
    }

    for v in 0..N{
        for d in 0..6{
            if g[v][d]==!0{
                g[v][d]=v;
            }
        }
    }

    let mut a=[!0;N];
    for i in 0..4{
        a[cand[i].0..cand[i].1].fill(i);
    }

    let mut v=tour[0];
    assert!(a[v]==obs[0]);
    for i in 0..TOUR{
        v=g[v][plan[i]];
        assert!(v==tour[i+1]);
        assert!(a[v]==obs[i+1]);
    }

    Some(Map{g,a,v:tour[0]})
}



#[derive(Clone)]
struct Map{
    g:[[usize;6];N],
    a:[usize;N],
    v:usize,
}
impl Map{
    fn gen(cnt:&[usize;4])->Map{
        let mut g=[[!0;6];N];
        let mut is=vec![];
        for i in 0..N{
            for j in 0..6{
                is.push((i,j));
            }
        }

        let rem=|is:&mut Vec<_>,a:(usize,usize)|{
            let idx=(0..is.len()).find(|&i|is[i]==a).unwrap();
            is.swap_remove(idx);
        };

        while !is.is_empty(){
            let (i0,j0)=is.choice();
            let (i1,j1)=is.choice();

            g[i0][j0]=i1;
            g[i1][j1]=i0;

            rem(&mut is,(i0,j0));
            if (i0,j0)!=(i1,j1){
                rem(&mut is,(i1,j1));
            }
        }

        let mut idx=0;
        let mut a=[!0;N];
        for i in 0..4{
            a[idx..idx+cnt[i]].fill(i);
            idx+=cnt[i];
        }
        rnd::shuffle(&mut a);

        let v=rnd::get(N);

        Map{g,a,v}
    }
    
    fn explore(&self,plan:&[[usize;TOUR]])->Vec<[usize;TOUR+1]>{
        let mut res=vec![];
        for p in plan{
            let mut obs=[!0;TOUR+1];
            let mut v=self.v;
            obs[0]=self.a[v];
            for i in 0..TOUR{
                v=self.g[v][p[i]];
                obs[i+1]=self.a[v];
            }
            res.push(obs);
        }
        res
    }
    
    fn is_isomorphic(&self,a:&Map)->bool{
        let mut it=4..;
        let mut set=std::collections::HashMap::new();
        let mut get=|a:[usize;7]|->usize{
            *set.entry(a).or_insert_with(||it.next().unwrap())
        };

        let mut dp0=self.a;
        let mut dp1=a.a;

        let mut nei=[!0;7];
        
        for _ in 0..N{
            for (g,dp) in [(&self.g,&mut dp0),(&a.g,&mut dp1)]{
                let mut new=[!0;N];
                for i in 0..N{
                    for j in 0..6{
                        nei[j]=dp[g[i][j]];
                    }
                    nei[6]=dp[i];
                    new[i]=get(nei);
                }
                *dp=new;
            }
        }

        dp0[self.v]==dp1[a.v]
    }

    fn isok(&self,a:&Map)->bool{
        let mut a0=self.a;
        let mut a1=a.a;

        let mut v0=self.v;
        let mut v1=a.v;

        for _ in 0..1e5 as usize{
            if a0[v0]!=a1[v1]{
                return false;
            }
            let new=rnd::get(4);
            a0[v0]=new;
            a1[v1]=new;
            let d=rnd::get(6);
            v0=self.g[v0][d];
            v1=a.g[v1][d];
        }
        
        true
    }
}



#[allow(unused)]
mod rnd {
    static mut X2:u32=12345;
    static mut X3:u32=0xcafef00d;
    static mut C_X1:u64=0xd15ea5e5<<32|23456;

    pub fn next()->u32{
        unsafe{
            let x=X3 as u64*3487286589;
            let ret=(X3^X2)+(C_X1 as u32^(x>>32) as u32);
            X3=X2;
            X2=C_X1 as u32;
            C_X1=x+(C_X1>>32);
            ret
        }
    }

    pub fn next64()->u64{
        (next() as u64)<<32|next() as u64
    }

    pub fn nextf()->f64{
        f64::from_bits(0x3ff0000000000000|(next() as u64)<<20)-1.
    }

    pub fn get(n:usize)->usize{
        assert!(0<n && n<=u32::MAX as usize);
        next() as usize*n>>32
    }

    pub fn range(a:usize,b:usize)->usize{
        assert!(a<b);
        get(b-a)+a
    }

    pub fn range_skip(a:usize,b:usize,skip:usize)->usize{
        assert!(a<=skip && skip<b);
        let n=range(a,b-1);
        n+(skip<=n) as usize
    }

    pub fn rangei(a:i64,b:i64)->i64{
        assert!(a<b);
        get((b-a) as usize) as i64+a
    }

    pub fn shuffle<T>(a:&mut [T]){
        for i in (1..a.len()).rev(){
            a.swap(i,get(i+1));
        }
    }

    pub fn shuffle_iter<T:Copy>(a:&mut [T])->impl Iterator<Item=T>+'_{
        (0..a.len()).rev().map(|i|{
            a.swap(i,get(i+1));
            a[i]
        })
    }
}


trait RandomChoice{
    type Output;
    fn choice(&self)->Self::Output;
}
impl<T:Copy> RandomChoice for [T]{
    type Output=T;
    fn choice(&self)->T{
        self[rnd::get(self.len())]
    }
}



#[macro_export]
macro_rules! static_data{
    ($a:ident:$t:ty|$e:expr)=>{
        fn $a()->&'static $t{
            use std::sync::OnceLock;
            #[allow(non_upper_case_globals)]
            static $a:OnceLock<$t>=OnceLock::new();
            $a.get_or_init(||$e)
        }
    }
}



fn logs()->&'static [f64;65536]{
    use std::sync::OnceLock;
    static LOGS:OnceLock<[f64;65536]>=OnceLock::new();
    LOGS.get_or_init(||{
        let mut log=[0.;65536];
        for i in 0..65536{
            log[i]=((i as f64+0.5)/65536.).ln();
        }
        rnd::shuffle(&mut log);
        log
    })
}


fn get_time()->f64{
    static mut START:f64=0.;
    let time=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();
    unsafe{
        if START==0.{
            START=time;
        }
        time-START
    }
}



// chatGPTが実装してくれた


use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, fs, path::Path};


impl Map {
    fn guess(&self) -> bool {
        match guess_impl(self) {
            Ok(ok) => ok,
            Err(err) => {
                eprintln!("[guess] error: {err:?}");
                false
            }
        }
    }
}

fn select_problem(name: &str) {
    if let Err(err) = select_problem_impl(name) {
        eprintln!("[select_problem] error: {err:?}");
        panic!();
    }
}

fn explore(plans: &[[usize; TOUR]]) -> Vec<[usize; TOUR + 1]> {
    match explore_impl(plans) {
        Ok(v) => v,
        Err(err) => {
            eprintln!("[explore] error: {err:?}");
            panic!();
        }
    }
}


const BASE_URL: &str = "https://31pwr5t6ij.execute-api.eu-west-2.amazonaws.com";

fn http() -> Client {
    Client::builder()
        .user_agent("aedificium-client/0.1")
        .build()
        .expect("failed to build reqwest client")
}

fn read_id() -> Result<String> {
    let p = Path::new(".id");
    let s = fs::read_to_string(p).with_context(|| format!("failed to read {:?}", p))?;
    Ok(s.trim().to_string())
}


#[derive(Serialize)]
struct SelectReq<'a> {
    id: &'a str,
    #[serde(rename = "problemName")]
    problem_name: &'a str,
}

#[derive(Deserialize)]
struct SelectResp {
    #[serde(rename = "problemName")]
    problem_name: String,
}

fn select_problem_impl(name: &str) -> Result<()> {
    let id = read_id()?;
    let body = SelectReq {
        id: &id,
        problem_name: name,
    };
    let url = format!("{BASE_URL}/select");
    let _resp: SelectResp = http().post(&url).json(&body).send()?.error_for_status()?.json()?;
    Ok(())
}


#[derive(Serialize)]
struct ExploreReq<'a> {
    id: &'a str,
    plans: Vec<String>,
}

#[derive(Deserialize)]
struct ExploreResp {
    results: Vec<Vec<i32>>,
    #[allow(dead_code)]
    #[serde(rename = "queryCount")]
    query_count: i64,
}

fn explore_impl(plans: &[[usize; TOUR]]) -> Result<Vec<[usize; TOUR + 1]>> {
    let id = read_id()?;

    let to_string_plan = |p: &[usize; TOUR]| -> String {
        let mut s = String::new();
        for &d in p{
            s.push(('0' as u8+d as u8) as char);
        }
        s
    };
    let plans_as_str: Vec<String> = plans.iter().map(to_string_plan).collect();

    let url = format!("{BASE_URL}/explore");
    let req = ExploreReq {
        id: &id,
        plans: plans_as_str,
    };
    let resp: ExploreResp = http().post(&url).json(&req).send()?.error_for_status()?.json()?;

    let mut out = Vec::with_capacity(resp.results.len());
    for rec in resp.results {
        let mut buf = [!0; TOUR + 1];
        for i in 0..TOUR+1{
            buf[i]=rec[i] as usize;
        }
        out.push(buf);
    }
    Ok(out)
}


#[derive(Serialize)]
struct GuessReq {
    id: String,
    map: MapOut,
}

#[derive(Serialize)]
struct MapOut {
    rooms: Vec<i32>,
    #[serde(rename = "startingRoom")]
    starting_room: i32,
    connections: Vec<Edge>,
}

#[derive(Serialize)]
struct EdgeEnd {
    room: i32,
    door: i32,
}

#[derive(Serialize)]
struct Edge {
    from: EdgeEnd,
    to: EdgeEnd,
}

#[derive(Deserialize)]
struct GuessResp {
    correct: bool,
}

fn guess_impl(m: &Map) -> Result<bool> {
    let id = read_id()?;

    let rooms: Vec<i32> = m.a.iter().map(|&t|t as i32).collect();
    let starting_room = m.v as i32;

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut connections = Vec::new();

    for r in 0..N {
        for d in 0..6 {
            if visited.contains(&(r, d)) {
                continue;
            }
            let to = m.g[r][d];
            let mut od = d;
            for od_try in 0..6 {
                if m.g[to][od_try] == r {
                    od = od_try;
                    break;
                }
            }

            visited.insert((r, d));
            visited.insert((to, od));

            connections.push(Edge {
                from: EdgeEnd {
                    room: r as i32,
                    door: d as i32,
                },
                to: EdgeEnd {
                    room: to as i32,
                    door: od as i32,
                },
            });
        }
    }

    let body = GuessReq {
        id,
        map: MapOut {
            rooms,
            starting_room,
            connections,
        },
    };

    let url = format!("{BASE_URL}/guess");
    let resp: GuessResp = http().post(&url).json(&body).send()?.error_for_status()?.json()?;
    Ok(resp.correct)
}