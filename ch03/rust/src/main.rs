use rand::prelude::*;
use rand_distr::Normal;
use std::sync::{Arc, Mutex}; 

use curve_fitting::vertex::*; 
use curve_fitting::edge::*;
use curve_fitting::problem::*;  
use curve_fitting::nalgebra_types::*; 

fn main() {
    let a = 1.0; 
    let b = 2.0; 
    let c = 1.0; 
    let n = 100; 
    let mean = 0.0;
    let std_dev = 0.2;
    let mut rng = thread_rng();
    let normal = Normal::new(mean, std_dev).expect("Invalid parameters for normal distribution");

    let mut problem = Problem::new(ProblemType::GenericProblem); 
    let mut vertex = CurveFittingVertex::new(3, 3);
    vertex.set_parameters(VecX::zeros(3));  
    let vertex = Arc::new(Mutex::new(vertex)); 
    problem.add_vertex(vertex.clone()); 

    for i in 0..n {
        let x = (i as f64) / 100.0;
        let noise_value = normal.sample(&mut rng);
        let y = f64::exp(a * x * x + b * x + c) + noise_value; 

        let mut edge = CurveFittingEdge::new(1, 1, vec!["CurveFittingEdge".to_string()], x, y);
        let edge_vertex = vec![vertex.clone()];
        edge.set_vertex(edge_vertex); 

        problem.add_edge(Arc::new(Mutex::new(edge))); 
    }

    problem.solve(30); 
    println!("-------After optimization, we got these parameters:");
    println!("{}", vertex.clone().lock().unwrap().parameters()); 
    println!("-------ground truth:"); 
    println!("1.0,  2.0,  1.0"); 
}
