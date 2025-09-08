#![allow(non_snake_case)]


const N:usize=12;
const TOUR:usize=N*6*2;
const K:usize=2;

const ITER:usize=1e7 as usize;
const DEBUG:bool=false;


// idを特定するコード、新たな出力部分、シュミレーション
// solve_cpp,solve_tourが全部の辺、頂点を訪れられる出力をしているかチェックする


fn main(){
    let mut cnt=[0;4];
    for i in 0..N{
        cnt[i%4]+=1;
    }

    run(&cnt,"daleth");
}



fn run(cnt:&[usize;4],name:&str){
    for _ in 0..100{
        eprintln!();
        eprintln!("explore-start");
        select_problem(name);
        let plan=make_plan();
        let obs=explore(&plan);
        eprintln!("explore-end");
        
        if let Some(ans)=solve(&cnt,&plan,&obs){
            if let Some((v,g,a))=adjust_graph_double(&ans){
                let f=guess(v,&g,&a);
                if f{
                    eprintln!("correct");
                    break;
                } else{
                    eprintln!("wrong");
                    continue;
                }
            }
        }
        eprintln!("giveup");
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
        let obs=map.explore(&plan);

        let res=solve(&cnt,&plan,&obs);
        eprint!("{} / {} : ",t,ty);
        
        if let Some(ans)=res{
            // let (v,g,a)=adjust_graph_double(&ans).unwrap();
            
            if map.is_isomorphic(&ans){
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
    eprintln!("wrong   : {:.2} ({wrong} / {ty})",wrong as f64/ty as f64);
    eprintln!("giveup  : {:.2} ({giveup} / {ty})",giveup as f64/ty as f64);
    eprintln!("time    : {:.2} sec",(end-start)/ty as f64);
}


#[derive(Clone,Copy)]
enum Act{
    Change(usize),
    Move(usize),
}
use Act::*;


fn adjust_graph_double(map:&Map)->Option<(usize,[[usize;6];N*2],[usize;N*2])>{
    let start=map.v;
    let g=&map.g;
    let a=&map.a;
    
    let g=restore_graph(g);
    let (cpp_start,_,tour_es)=solve_cpp(&g,start);
    let tour_vs=solve_tsp(&g,start,&[cpp_start]);

    let mut shuf=[[0;6];N];
    for i in 0..N{
        for j in 0..6{
            let new=rnd::get(2);
            shuf[i][j]=new;
            let (a,b)=g[i][j];
            shuf[a][b]=new;
        }
    }
    let gstart=start+if rnd::get(2)==0{N}else{0};

    let explore=|plan:&[Act]|->Vec<usize>{
        let mut v=gstart;
        let mut a={
            let mut new=[!0;N*2];
            for i in 0..N{
                new[i]=a[i];
                new[i+N]=a[i];
            }
            new
        };

        let mut obs=vec![a[v]];
        for &act in plan{
            match act{
                Change(x)=>{
                    a[v]=x;
                },
                Move(d)=>{
                    let mut nv=g[v%N][d].0;
                    if (v/N)^(nv/N)!=shuf[v%N][d]{
                        if nv<N{
                            nv+=N;
                        } else{
                            nv-=N;
                        }
                    }
                    assert!((v/N)^(nv/N)==shuf[v%N][d]);
                    v=nv;
                },
            }
            obs.push(a[v]);
        }

        obs
    };

    // let explore=|plan:&[Act]|->Vec<usize>{
    //     eprintln!("explore-start");
    //     let ans=explore_with_act(&[plan.to_vec()]);
    //     eprintln!("explore-end");
    //     ans[0].to_vec()
    // };

    let mut seen=[false;N];
    let mut v=start;
    seen[v]=true;
    
    let mut plan=vec![Change(a[v]^1)];

    for &d in &tour_vs{
        plan.push(Move(d));

        v=g[v][d].0;
        if !seen[v]{
            seen[v]=true;
            plan.push(Change(a[v]^1));
        }
    }

    assert!(seen==[true;N]);
    let mut seen=[[false;6];N];
    let mut info=vec![];
    
    for &d in &tour_es{
        let (nv,nd)=g[v][d];
        plan.push(Move(d));
        
        if !seen[v][d]{
            seen[v][d]=true;
            seen[nv][nd]=true;
            info.push((plan.len()-1,plan.len(),(v,d),(nv,nd)));
        }
        v=nv;
    }

    let movs=plan.iter().filter(|&act|matches!(act,Move(_))).count();
    assert!(movs<=TOUR);

    assert!(seen==[[true;6];N]);
    let obs=explore(&plan);
    assert!(obs.len()==plan.len()+1);

    let mut newa=[!0;N*2];
    for i in 0..N{
        newa[i]=a[i];
        newa[i+N]=a[i];
    }
    let new_start=start+N;
    
    let mut newg=[[!0;6];N*2];
    
    for &(i0,i1,(v0,d0),(v1,d1)) in &info{
        let o0=obs[i0];
        let o1=obs[i1];

        if !(o0==a[v0] || o0==a[v0]^1){
            return None;
        }
        if !(o1==a[v1] || o1==a[v1]^1){
            return None;
        }
        // assert!(o0==a[v0] || o0==a[v0]^1);
        // assert!(o1==a[v1] || o1==a[v1]^1);

        let f0=o0==a[v0];
        let f1=o1==a[v1];

        if f0==f1{
            newg[v0][d0]=v1;
            newg[v1][d1]=v0;
            newg[v0+N][d0]=v1+N;
            newg[v1+N][d1]=v0+N;
        } else{
            newg[v0][d0]=v1+N;
            newg[v1][d1]=v0+N;
            newg[v0+N][d0]=v1;
            newg[v1+N][d1]=v0;
        }
    }

    assert!(newa[new_start]==obs[0]);
    {
        let mut v=new_start;
        let mut a=newa;
        assert!(a[v]==obs[0]);
        for i in 0..plan.len(){
            match plan[i]{
                Change(x)=>{
                    a[v]=x;
                },
                Move(d)=>{
                    v=newg[v][d];
                },
            }
            if a[v]!=obs[i+1]{
                return None;
            }
            // assert!(a[v]==obs[i+1]);
        }
    }

    Some((new_start,newg,newa))
}



fn adjust_graph_triple(map:&Map)->(usize,[[usize;6];N*3],[usize;N*3]){
    let start=map.v;
    let g=&map.g;
    let a=&map.a;

    let g=restore_graph(g);
    
    todo!();
}



fn make_plan()->[[usize;TOUR];K]{
    // assert!(K<6);
    let gen=||{
        let mut it=0..;
        let mut plan=[();TOUR].map(|_|it.next().unwrap()%6);
        rnd::shuffle(&mut plan);
        plan
    };

    let mut plan=[[!0;TOUR];K];
    for i in 0..K{
        plan[i]=gen();
        // loop{
        //     plan[i]=gen();
        //     if plan[i][0]==i{
        //         break;
        //     }
        // }
    }
    plan
}



fn solve(cnt:&[usize;4],plans:&[[usize;TOUR];K],obss:&[[usize;TOUR+1];K])->Option<Map>{
    const L:usize=TOUR*K+1;
    
    let mut plan=[!0;L];
    let mut obs=[!0;L];
    obs[0]=obss[0][0];
    for i in 0..K{
        assert!(obss[i][0]==obs[0]);
        for j in 0..TOUR{
            plan[i*TOUR+j+1]=plans[i][j];
            obs[i*TOUR+j+1]=obss[i][j+1];
        }
    }

    let get_prev=|i:usize|->usize{
        assert!(i!=0);
        if i%TOUR==1{
            0
        } else{
            i-1
        }
    };
    let get_next=|i:usize|->Option<usize>{
        assert!(i!=0);
        if i%TOUR==0{
            None
        } else{
            Some(i+1)
        }
    };

    // obsがiの頂点idの候補はcand[i].0..cand[i].1
    let mut cand=[(0,cnt[0]);4];
    for i in 1..4{
        cand[i]=(cand[i-1].1,cand[i-1].1+cnt[i]);
    }

    let is_only=|o:usize|->bool{
        cand[o].1-cand[o].0==1
    };
    let gen=|o:usize|{
        rnd::range(cand[o].0,cand[o].1)
    };
    let gen_skip=|o:usize,skip:usize|{
        rnd::range_skip(cand[o].0,cand[o].1,skip)
    };

    let mut tour=[!0;L];
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
    for i in 1..L{
        cnt[tour[get_prev(i)]][plan[i]][tour[i]]+=1;
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

        let i=rnd::get(L);
        let cv=tour[i];

        // todo : 接続を維持する近傍を優先してる、これ大丈夫なのか怪しい
        let nv={
            let mut r=(0,0);
            let o=obs[i];
            if is_only(o){
                continue;
            }
            let mut res=gen_skip(o,cv);

            if i==0{
                for v in cand[o].0..cand[o].1{
                    let mut f=0;
                    for ni in (1..tour.len()).step_by(TOUR){
                        f+=(cnt[v][plan[ni]][tour[i]]>0) as u8;
                    }

                    if v!=cv && f>0{
                        let new=(f,rnd::next());
                        if r<new{
                            r=new;
                            res=v;
                        }
                    }
                }
            } else if let Some(ni)=get_next(i){
                let pi=get_prev(i);
                for v in cand[o].0..cand[o].1{
                    let f=(cnt[tour[pi]][plan[i]][v]>0) as u8+(cnt[v][plan[ni]][tour[ni]]>0) as u8;
                    if v!=cv && f>0{
                        let new=(f,rnd::next());
                        if r<new{
                            r=new;
                            res=v;
                        }
                    }
                }
            } else{
                let pi=get_prev(i);
                for v in cand[o].0..cand[o].1{
                    if v!=cv && cnt[tour[pi]][plan[i]][v]>0{
                        let new=(1,rnd::next());
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

        if i==0{
            for ni in (1..tour.len()).step_by(TOUR){
                let next=tour[ni];
                let nd=plan[ni];

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
        } else{
            let pi=get_prev(i);
            let prev=tour[pi];
            let pd=plan[i];
    
            let (arg,add)=get(&cnt[prev][pd]);
            new-=add;
            new+=add_es!(prev,arg,-1);
            
            cnt[prev][pd][cv]-=1;
            cnt[prev][pd][nv]+=1;
    
            let (arg,add)=get(&cnt[prev][pd]);
            new+=add;
            new+=add_es!(prev,arg,1);
            
            if let Some(ni)=get_next(i){
                let next=tour[ni];
                let nd=plan[ni];

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

            if i==0{
                for ni in (1..tour.len()).step_by(TOUR){
                    let next=tour[ni];
                    let nd=plan[ni];

                    cnt[cv][nd][next]+=1;
                    cnt[nv][nd][next]-=1;
                }
            } else{
                let pi=get_prev(i);
                let prev=tour[pi];
                let pd=plan[i];

                cnt[prev][pd][cv]+=1;
                cnt[prev][pd][nv]-=1;

                if let Some(ni)=get_next(i){
                    let next=tour[ni];
                    let nd=plan[ni];

                    cnt[cv][nd][next]+=1;
                    cnt[nv][nd][next]-=1;
                }
            }

            for &(a,b,c) in &hi{
                es[a][b]-=c;
            }
        }
    }

    {
        let mut verify_cnt=[[[0;N];6];N];
        for i in 1..L{
            verify_cnt[tour[get_prev(i)]][plan[i]][tour[i]]+=1;
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

    assert!(a[tour[0]]==obs[0]);
    for i in 1..L{
        let pi=get_prev(i);
        assert!(g[tour[pi]][plan[i]]==tour[i]);
        assert!(a[tour[i]]==obs[i]);
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
    
    fn explore(&self,plan:&[[usize;TOUR];K])->[[usize;TOUR+1];K]{
        let mut obs=[[!0;TOUR+1];K];
        for k in 0..K{
            let p=&plan[k];
            let mut v=self.v;
            obs[k][0]=self.a[v];
            for i in 0..TOUR{
                v=self.g[v][p[i]];
                obs[k][i+1]=self.a[v];
            }
        }
        obs
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
        match guess_impl(self.v,&self.g,&self.a) {
            Ok(ok) => ok,
            Err(err) => {
                eprintln!("[guess] error: {err:?}");
                false
            }
        }
    }
}


fn guess(v:usize,g:&[[usize;6]],a:&[usize])->bool{
    match guess_impl(v,g,a) {
        Ok(ok) => ok,
        Err(err) => {
            eprintln!("[guess] error: {err:?}");
            false
        }
    }
}


fn select_problem(name: &str) {
    if let Err(err) = select_problem_impl(name) {
        eprintln!("[select_problem] error: {err:?}");
        panic!();
    }
}

fn explore(plans: &[[usize; TOUR];K]) -> [[usize; TOUR + 1];K] {
    let mut a=vec![vec![];K];
    for i in 0..K{
        a[i]=plans[i].iter().map(|&t|Move(t)).collect();
    }
    
    match explore_impl(&a) {
        Ok(v) => {
            let mut obss=[[!0;TOUR+1];K];
            assert!(v.len()==K);
            for i in 0..K{
                assert!(v[i].len()==TOUR+1);
                for j in 0..TOUR+1{
                    obss[i][j]=v[i][j];
                }
            }
            obss
        },
        Err(err) => {
            eprintln!("[explore] error: {err:?}");
            panic!();
        }
    }
}


fn explore_with_act(plans:&[Vec<Act>])->Vec<Vec<usize>>{
    match explore_impl(plans){
        Ok(v)=>v,
        Err(err)=>{
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

fn explore_impl(plans:&[Vec<Act>]) -> Result<Vec<Vec<usize>>> {
    let id = read_id()?;

    let to_char=|d:usize|{
        assert!(d<9);
        ('0' as u8+d as u8) as char
    };
    let to_string = |p: &[Act]| -> String {
        let mut s = String::new();
        for &act in p{
            match act{
                Change(x)=>{
                    s.push('[');
                    s.push(to_char(x));
                    s.push(']');
                },
                Move(d)=>{
                    s.push(to_char(d));
                },
            }
        }
        s
    };
    let plans_as_str=plans.iter().map(|a|to_string(a)).collect();

    let url = format!("{BASE_URL}/explore");
    let req = ExploreReq {
        id: &id,
        plans: plans_as_str,
    };
    let resp: ExploreResp = http().post(&url).json(&req).send()?.error_for_status()?.json()?;

    let mut obss=vec![vec![];plans.len()];
    assert!(resp.results.len() == plans.len());
    for i in 0..plans.len(){
        obss[i]=resp.results[i].iter().map(|&t|t as usize).collect();
    }
    Ok(obss)
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

fn guess_impl(v:usize,g:&[[usize;6]],a:&[usize]) -> Result<bool> {
    let id = read_id()?;

    let rooms: Vec<i32> = a.iter().map(|&t|t as i32).collect();
    let starting_room = v as i32;

    let n=g.len();
    let mut cand=vec![vec![vec![];n];n];
    for v in 0..n{
        for d in 0..6{
            cand[v][g[v][d]].push((v,d));
        }
    }

    let mut ret=vec![[(!0,!0);6];n];
    for v in 0..n{
        for d in 0..6{
            let nv=g[v][d];
            if nv==v{
                ret[v][d]=(v,d);
            } else{
                ret[v][d]=cand[nv][v].pop().unwrap();
                assert!(nv==ret[v][d].0);
            }
        }
    }

    let g=ret;

    let mut connections = vec![];
    for v in 0..g.len() {
        for d in 0..6 {
            let (nv,nd)=g[v][d];
            if (v,d)<=(nv,nd){
                connections.push(Edge {
                    from: EdgeEnd {
                        room: v as i32,
                        door: d as i32,
                    },
                    to: EdgeEnd {
                        room: nv as i32,
                        door: nd as i32,
                    },
                });
            }
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


const INF:i64=i64::MAX/100;



fn restore_graph(g:&[[usize;6];N])->[[(usize,usize);6];N]{
    let mut cand=vec![vec![vec![];N];N];
    for v in 0..N{
        for d in 0..6{
            cand[v][g[v][d]].push((v,d));
        }
    }

    // let mut self_loop=0;
    let mut ret=[[(!0,!0);6];N];
    for v in 0..N{
        for d in 0..6{
            let nv=g[v][d];
            if nv==v{
                // self_loop+=1;
                ret[v][d]=(v,d);
            } else{
                ret[v][d]=cand[nv][v].pop().unwrap();
                assert!(nv==ret[v][d].0);
            }
        }
    }
    // eprintln!("edges = {}",(6*N+self_loop)/2);

    ret
}



fn get_odds(g:&[[(usize,usize);6];N])->Vec<usize>{
    let mut deg=[0;N];
    for i in 0..N{
        for j in 0..6{
            if g[i][j].0!=i{
                deg[i]+=1;
            }
        }
    }
    (0..N).filter(|&i|deg[i]%2==1).collect()
}



fn solve_apsp(g:&[[(usize,usize);6];N])->([[i64;N];N],[[(usize,usize);N];N]){
    let mut d=[[INF;N];N];
    let mut next=[[(!0,!0);N];N];
    for i in 0..N{
        d[i][i]=0;
    }
    for i in 0..N{
        for j in 0..6{
            d[i][g[i][j].0]=1;
            next[i][g[i][j].0]=(j,g[i][j].0);
        }
    }
    for k in 0..N{
        for i in 0..N{
            for j in 0..N{
                let new=d[i][k]+d[k][j];
                if d[i][j]>new{
                    d[i][j]=new;
                    next[i][j]=next[i][k];
                }
            }
        }
    }
    (d,next)
}



fn solve_tsp(g:&[[(usize,usize);6];N],start:usize,end:&[usize])->Vec<usize>{
    assert!(!end.contains(&start));

    let (di,next)=solve_apsp(g);
    let mut d=vec![vec![INF;N+1];N+1];
    for i in 0..N{
        for j in 0..N{
            d[i][j]=di[i][j];
        }
    }
    
    d[N][N]=0;
    d[start][N]=1;
    d[N][start]=1;
    for &v in end{
        d[v][N]=10000;
        d[N][v]=10000;
    }
    for v in 0..N{
        if d[N][v]!=INF{
            d[v][N]=10000000;
            d[N][v]=10000000;
        }
    }

    let tour=tsp::greedy(&d,start);
    let tour=tsp::lis(&d,&tour,get_time()+0.1);

    assert!(tour[1]==N || tour[N]==N);
    let tour=if tour[1]==N{
        tour[2..].iter().rev().copied().collect()
    } else{
        tour[..N].to_vec()
    };

    assert!(tour.len()==N);
    assert!(tour[0]==start);
    assert!(end.contains(&tour[N-1]));
    
    let mut v=start;
    let mut res=vec![];
    for &nv in &tour[1..]{
        while v!=nv{
            res.push(next[v][nv].0);
            v=next[v][nv].1;
        }
    }

    res
}



// 参考：https://atcoder.jp/contests/tessoku-book/submissions/34976398
// 巡回ではなくパスにしたいなら適当な番兵を1つ置けば、始点固定、終点固定、始点終点自由のすべての場合でどうにかなる
#[allow(unused)]
mod tsp{
    use super::*;

    type C=i64;
    const EPS:C=0;
    // const EPS:C=1e-5;

    pub fn get_cost(g:&Vec<Vec<C>>,ps:&[usize])->C{
        (1..ps.len()).map(|i|g[ps[i-1]][ps[i]]).sum()
    }

    pub fn greedy(g:&Vec<Vec<C>>,start:usize)->Vec<usize>{
        let mut tour=vec![start];
        let mut rem=(0..g.len()).filter(|&v|v!=start).collect::<Vec<_>>();
        let mut v=start;

        for _ in 0..rem.len(){
            let arg=(0..rem.len()).min_by_key(|&i|g[v][rem[i]]).unwrap();
            v=rem.swap_remove(arg);
            tour.push(v);
        }

        tour.push(start);
        tour
    }

    // mv: (i, dir)
    // なにこれ？
    fn apply_move(tour:&mut [usize],idx:&mut [usize],mv:&[(usize,usize)]){
        let k=mv.len();
        let mut ids=(0..k).collect::<Vec<_>>();
        ids.sort_unstable_by_key(|&i|mv[i].0);
        
        let mut order=vec![0;k];
        for i in 0..k{
            order[ids[i]]=i;
        }

        let mut tour2=Vec::with_capacity(mv[ids[k-1]].0-mv[ids[0]].0);
        let mut i=ids[0];
        let mut dir=0;

        loop{
            let (j,rev)=if dir==mv[i].1{
                ((i+1)%k,0)
            } else{
                ((i+k-1)%k,1)
            };
            if mv[j].1==rev{
                if order[j]==k-1{
                    break;
                }
                i=ids[order[j]+1];
                dir=0;
                tour2.extend_from_slice(&tour[mv[j].0+1..mv[i].0+1]);
            } else {
                i=ids[order[j]-1];
                dir=1;
                tour2.extend(tour[mv[i].0+1..mv[j].0+1].iter().rev().copied());
            }
        }
        assert!(tour2.len()==mv[ids[k-1]].0-mv[ids[0]].0);

        tour[mv[ids[0]].0+1..mv[ids[k-1]].0+1].copy_from_slice(&tour2);
        for i in mv[ids[0]].0+1..mv[ids[k-1]].0+1{
            idx[tour[i]]=i;
        }
    }

    const FEASIBLE3:u64=0b1011111101000000000000001011111101000000000000001011111101000;

    pub fn lis(g:&Vec<Vec<C>>,qs:&[usize],tl:f64)->Vec<usize>{
        let n=g.len();
        for u in 0..n{
            for v in 0..u{
                assert!(g[u][v]==g[v][u]);
            }
        }

        let mut f=vec![vec![];n];
        for i in 0..n{
            f[i]=(0..n).filter(|&j|i!=j).map(|j|(g[i][j],j)).collect();
            f[i].sort_unstable_by_key(|a|a.0);
        }

        let mut ps=qs.to_vec();
        let mut idx=vec![!0;n];
        let mut min=get_cost(&g,qs);
        let mut min_ps=ps.clone();

        while get_time()<tl{
            let mut cost=get_cost(&g,&ps);
            for p in 0..n{
                idx[ps[p]]=p;
            }

            loop{
                let mut ok=false;
                for i in 0..n{
                    for di in 0..2{
                        'a: for &(ij,vj) in &f[ps[i+di]]{
                            if g[ps[i]][ps[i+1]]-ij<=EPS{
                                break;
                            }

                            for dj in 0..2{
                                let j=if idx[vj]==0 && dj==0{
                                    n-1
                                } else {
                                    idx[vj]-1+dj
                                };

                                let gain=g[ps[i]][ps[i+1]]-ij+g[ps[j]][ps[j+1]];
                                let diff=g[ps[j+dj]][ps[i+1-di]]-gain;
                                
                                // 2-opt
                                if di!=dj && diff < -EPS{
                                    cost+=diff;
                                    apply_move(&mut ps,&mut idx,&[(i,di),(j,dj)]);
                                    ok=true;
                                    break 'a;
                                }

                                // 3-opt
                                for &(jk,vk) in &f[ps[j+dj]]{
                                    if gain-jk<=EPS{
                                        break;
                                    }

                                    for dk in 0..2{
                                        let k=if idx[vk]==0 && dk==0{
                                            n-1
                                        } else{
                                            idx[vk]-1+dk
                                        };

                                        if i==k || j==k{
                                            continue;
                                        }

                                        let diff=g[ps[k+dk]][ps[i+1-di]]-gain+jk-g[ps[k]][ps[k+1]];
                                        if diff < -EPS{
                                            let mask=((i<j) as usize)<<5 | ((i<k) as usize)<<4 |
                                                ((j<k) as usize)<<3 | di<<2 | dj<<1 | dk;

                                            if FEASIBLE3>>mask&1==1{
                                                cost+=diff;
                                                apply_move(&mut ps,&mut idx,&[(i,di),(j,dj),(k,dk)]);
                                                ok=true;
                                                break 'a;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if !ok{
                    break;
                }
            }

            if min>cost{
                min=cost;
                min_ps=ps;
            }
            ps=min_ps.clone();

            if rnd::next()&1==0{ // todo
                // double bridge
                let mut is=[!0;4];
                loop{
                    is.fill_with(||rnd::range(0,n));
                    is.sort_unstable();

                    if is[0]==is[1] || is[1]==is[2] || is[2]==is[3]{
                        continue;
                    }
                    break;
                }

                ps=ps[0..is[0]+1].iter()
                    .chain(&ps[is[2]+1..is[3]+1])
                    .chain(&ps[is[1]+1..is[2]+1])
                    .chain(&ps[is[0]+1..is[1]+1])
                    .chain(&ps[is[3]+1..])
                    .copied().collect();
            } else{
                for _ in 0..6{ // todo
                    let mut i=rnd::range(1,n);
                    let mut j=rnd::range_skip(1,n,i);
                    if i>j{
                        (i,j)=(j,i);
                    }
                    ps[i..j+1].reverse();
                }
            }
        }

        min_ps
    }
}



fn solve_cpp(g:&[[(usize,usize);6];N],skip:usize)->(usize,usize,Vec<usize>){
    let odds=get_odds(g);
    if odds.is_empty(){
        let mut es=vec![];
        let mut id=vec![];
        for i in 0..N{
            for j in 0..6{
                if i<=g[i][j].0{
                    es.push((i,g[i][j].0));
                    id.push(((i,j),g[i][j]));
                }
            }
        }
        let start=if skip==0{
            1
        } else{
            0
        };
        let path=find_eular_path(N,start,&es);

        let mut door=vec![];
        let mut v=start;
        for &i in &path{
            let ((v0,d0),(v1,d1))=id[i];
            assert!(v==v0 || v==v1);
            if v==v0{
                door.push(d0);
                v=v1;
            } else{
                door.push(d1);
                v=v0;
            }
        }

        return (start,start,door);
    }

    assert!(odds.len()%2==0);
    let (d,next)=solve_apsp(g);

    let mut mat=vec![!0;odds.len()];
    for i in 0..odds.len(){
        mat[i]=i^1;
    }
    let get_score=|mat:&[usize]|->i64{
        let mut score=0;
        let mut max=0;
        for i in 0..odds.len(){
            if i<mat[i]{
                let d=d[odds[i]][odds[mat[i]]];
                score+=d;
                max=max.max(d);
            }
        }
        score-max
    };
    let mut score=get_score(&mat);

    let mut valid=0;
    let log=logs();
    let tot=1e6 as usize;

    if odds.len()>2{
        for it in 0..tot{
            let T0:f64=10.;
            let T1:f64=0.;
            let time=it as f64/tot as f64;
            let heat=T0+(T1-T0)*time;

            let add=-heat*log[it&65535]; // 最小化

            let i0=rnd::get(odds.len());
            let i1=mat[i0];
            assert!(i0!=i1);

            let mut j0=rnd::get(odds.len()-2);
            if i0.min(i1)<=j0{
                j0+=1;
            }
            if i0.max(i1)<=j0{
                j0+=1;
            }
            let j1=mat[j0];

            mat[i0]=j0;
            mat[i1]=j1;
            mat[j0]=i0;
            mat[j1]=i1;

            let new=get_score(&mat);
            if (new-score) as f64<=add{ // 最小化
                score=new;
                valid+=1;
            } else{
                mat[i0]=i1;
                mat[i1]=i0;
                mat[j0]=j1;
                mat[j1]=j0;
            }
        }    
    }

    assert!(score==get_score(&mat));
    if DEBUG{
        eprintln!("ratio = {:.3}",valid as f64/tot as f64);
        eprintln!("score = {score}");
    }

    let mut es=vec![];
    let mut id=vec![];
    for i in 0..N{
        for j in 0..6{
            if i<=g[i][j].0{
                es.push((i,g[i][j].0));
                id.push(((i,j),g[i][j]));
            }
        }
    }

    let mut max=0;
    let mut arg=!0;
    for i in 0..odds.len(){
        if i<mat[i]{
            let d=d[odds[i]][odds[mat[i]]];
            if max<d{
                max=d;
                arg=i;
            }
        }
    }

    for i in 0..odds.len(){
        if i<mat[i] && i!=arg{
            let mut v=odds[i];
            let goal=odds[mat[i]];

            while v!=goal{
                let (d,nv)=next[v][goal];
                es.push((v,nv));
                id.push(((v,d),g[v][d]));
                v=nv;
            }
        }
    }

    let mut start=odds[arg];
    let mut goal=odds[mat[arg]];
    if start==skip{
        (start,goal)=(goal,start)
    }

    let path=find_eular_path(N,start,&es);

    let mut door=vec![];
    let mut v=start;
    for &i in &path{
        let ((v0,d0),(v1,d1))=id[i];
        assert!(v==v0 || v==v1);
        if v==v0{
            door.push(d0);
            v=v1;
        } else{
            door.push(d1);
            v=v0;
        }
    }
    
    (start,goal,door)
}



/// 無向グラフ（多重辺・自己ループ可）で、start から始まるオイラー路を
/// 「通過する辺 es の index の列」で返す。
/// グラフが start からのオイラー路条件を満たさない場合は assert で落とす。
fn find_eular_path(n: usize, start: usize, es: &[(usize, usize)]) -> Vec<usize> {
    assert!(start < n, "start is out of range");

    // 辺が無ければ空経路
    if es.is_empty() {
        return Vec::new();
    }

    // 隣接リスト: (to, edge_index)
    let mut adj = vec![Vec::<(usize, usize)>::new(); n];
    let mut deg = vec![0usize; n];

    for (i, &(a, b)) in es.iter().enumerate() {
        assert!(a < n && b < n, "edge {} has invalid endpoint(s)", i);
        // 無向なので両方向に格納（自己ループも2回入れる）
        adj[a].push((b, i));
        adj[b].push((a, i));
        if a == b {
            deg[a] += 2; // 自己ループは次数+2
        } else {
            deg[a] += 1;
            deg[b] += 1;
        }
    }

    // オイラー路存在条件（無向）
    // ・奇数次数の頂点数は 0 or 2
    // ・2 のときは start がその一方
    let odd: Vec<usize> = deg
        .iter()
        .enumerate()
        .filter_map(|(v, &d)| if d % 2 == 1 { Some(v) } else { None })
        .collect();
    assert!(
        odd.len() == 0 || odd.len() == 2,
        "No Euler trail: #odd = {}",
        odd.len()
    );
    if odd.len() == 2 {
        assert!(
            odd[0] == start || odd[1] == start,
            "Trail must start at an odd-degree vertex (odd={:?}, start={})",
            odd,
            start
        );
    }
    assert!(deg[start] > 0, "start has degree 0 while edges exist");

    // 連結性チェック（次数>0 の頂点が start から到達可能）
    let mut vis = vec![false; n];
    let mut stk = vec![start];
    vis[start] = true;
    while let Some(v) = stk.pop() {
        for &(to, _) in &adj[v] {
            if !vis[to] {
                vis[to] = true;
                stk.push(to);
            }
        }
    }
    for v in 0..n {
        if deg[v] > 0 {
            assert!(vis[v], "Graph is disconnected wrt start ({} unreachable)", v);
        }
    }

    // Hierholzer（非再帰）
    let mut it = vec![0usize; n];        // 各頂点での走査位置
    let mut used = vec![false; es.len()];// 辺使用フラグ（無向なので1本で共有）
    let mut stack_v: Vec<usize> = vec![start];
    // 各頂点に入るのに使った辺（最初は None）
    let mut stack_e: Vec<Option<usize>> = vec![None];
    let mut ans: Vec<usize> = Vec::with_capacity(es.len());

    while !stack_v.is_empty() {
        let v = *stack_v.last().unwrap();

        // まだ未使用の辺を探す
        while it[v] < adj[v].len() && used[adj[v][it[v]].1] {
            it[v] += 1;
        }

        if it[v] == adj[v].len() {
            // 行き止まり：この頂点に入るために使った辺を答えに積む
            let e_in = stack_e.pop().unwrap();
            stack_v.pop().unwrap();
            if let Some(ei) = e_in {
                ans.push(ei);
            }
        } else {
            let (to, ei) = adj[v][it[v]];
            it[v] += 1;
            if used[ei] {
                continue;
            }
            used[ei] = true;
            stack_v.push(to);
            stack_e.push(Some(ei));
        }
    }

    // すべての辺をちょうど一度ずつ使っているか
    assert!(
        ans.len() == es.len(),
        "Used {} / {} edges",
        ans.len(),
        es.len()
    );

    ans.reverse(); // 逆順で溜めたので反転して正順に
    ans
}



// 同じ辺を使わずにvからvに戻る経路
fn find_minimum_cycle(g:&[[(usize,usize);6];N],v:usize)->Option<Vec<usize>>{
    let find_path=|start:usize,goal:usize,skip0:(usize,usize),skip1:(usize,usize)|->Option<Vec<usize>>{
        let mut que=std::collections::VecDeque::from([start]);
        let mut seen=[false;N];
        seen[start]=true;
        let mut prev=[(!0,!0);N];

        while let Some(v)=que.pop_front(){
            if v==goal{
                break;
            }

            for d in 0..6{
                if (v,d)==skip0 || (v,d)==skip1{
                    continue;
                }

                let nv=g[v][d].0;
                if !seen[nv]{
                    seen[nv]=true;
                    prev[nv]=(v,d);
                    que.push_back(nv);
                }
            }
        }

        if !seen[goal]{
            return None;
        }

        let mut res=vec![];
        let mut v=goal;
        while v!=start{
            let (pv,d)=prev[v];
            res.push(d);
            v=pv;
        }
        res.reverse();
        Some(res)
    };

    let mut min=usize::MAX/100;
    let mut ans=vec![];

    for i in 0..6{
        if let Some(mut new)=find_path(v,g[v][i].0,(v,i),g[v][i]){
            new.push(g[v][i].1);
            if min>new.len(){
                min=new.len();
                ans=new;
            }
        }
    }

    if ans.is_empty(){
        None
    } else{
        Some(ans)
    }
}