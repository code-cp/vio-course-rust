use nalgebra as na; 
use rand::Rng; 

fn generate_unit_quaternion() -> na::UnitQuaternion<f64> {
    let mut rng = rand::thread_rng(); 
    let pi: f64 = std::f64::consts::PI; 

    let axis = na::Unit::new_normalize(
        na::Vector3::new(
            rng.gen(), 
            rng.gen(), 
            rng.gen(), 
        )
    ); 

    let angle: f64 = rng.gen(); 
    let angle: f64 = angle * pi; 
    let rotation = na::UnitQuaternion::from_axis_angle(
        &axis, angle, 
    ); 

    rotation 
}

/// slam book eq. 2.4 
fn skew(v: &na::Vector3<f64>) -> na::Matrix3<f64> {
    let mut ss = na::Matrix3::zeros();
    ss[(0, 1)] = -v[2];
    ss[(0, 2)] = v[1];
    ss[(1, 0)] = v[2];
    ss[(1, 2)] = -v[0];
    ss[(2, 0)] = -v[1];
    ss[(2, 1)] = v[0];
    ss
}

/// slam book eq 4.22 
fn exp_map(v: &na::Vector3<f64>) -> na::Rotation3<f64> {
    let theta = v.norm();
    let n = v / theta; 
    let r_mat = na::Matrix3::identity() * theta.cos() + (1.0 - theta.cos()) * n * n.transpose() + theta.sin() * skew(&n);  
    let r = na::Rotation3::from_matrix(&r_mat); 
    r
}

fn main() {
    let quaternion = generate_unit_quaternion(); 
    let rotation = na::Rotation3::from_axis_angle(
        &quaternion.axis().unwrap(), 
        quaternion.angle(), 
    ); 

    let omega = na::Vector3::new(0.01, 0.02, 0.03); 

    let rotation_updated = rotation * exp_map(&omega);
    // Note that the arguments order does not follow the storage order
    // ref https://docs.rs/nalgebra/latest/nalgebra/geometry/struct.Quaternion.html#method.new
    let quaternion_updated = quaternion.quaternion() * na::Quaternion::new(
        1.0, 
        0.5 * omega[0], 
        0.5 * omega[1], 
        0.5 * omega[2], 
    );
    let quaternion_updated = na::UnitQuaternion::from_quaternion(quaternion_updated); 

    println!("angle from rotation: {}", rotation_updated.angle()); 
    println!("angle from unit quaternion: {}", quaternion_updated.angle());
    assert_eq!(rotation_updated.angle(), quaternion_updated.angle());
}
