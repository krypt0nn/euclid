use super::prelude::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Information used for values backpropagation.
///
/// Stores AdamW optimizer info for `SIZE` parameters in `F` float type.
/// Uses cyclic learn rate scheduler with warmup stage.
///
/// If you plan to perform `N` backpropagation calls, then it's recommended to:
///
/// - Set `warmup_duration` from `0.05 * N` to `0.1 * N` (5-10% of total time).
/// - Set `cycle_duration` from `0.1 * N` to `0.2 * N` (10-20% of total time).
/// - Set `cycle_radius` from `3 * learn_rate` to `10 * learn_rate`.
///
/// This might seem weird to set large cycle radius, but its goal is to leave
/// the local minimum of the loss function for given parameters set, and large
/// learn rate changes are needed to achieve this.
///
/// It's not recommended to change AdamW optimizer values, defaults are found
/// to be most suitable for all the cases by many researchers.
pub struct Backpropagation<const SIZE: usize, F: Float> {
    /// Current backpropagation timestep.
    ///
    /// Shows amount of already passed backpropagations.
    timestep: u32,

    /// AdamW backpropagation optimization momentums.
    adamw_m: Box<[F]>,

    /// AdamW backpropagation optimization squared momentums.
    adamw_v: Box<[F]>,

    /// AdamW backpropagation optimization hyperparameter.
    ///
    /// Beta 1 controls the exponential moving average of the gradient.
    /// A high value of beta 1 means that the optimizer will give more weight
    /// to the previous gradients, while a low value means that the optimizer
    /// will give more weight to the current gradient.
    ///
    /// Recommended values: 0.9, 0.95, 0.99.
    adamw_beta1: F,

    /// AdamW backpropagation optimization hyperparameter.
    ///
    /// Beta 2 controls the exponential moving average of the squared gradient.
    /// A high value of beta 2 means that the optimizer will give more weight
    /// to the previous squared gradients, while a low value means that the optimizer
    /// will give more weight to the current squared gradient.
    ///
    /// Recommended values: 0.999, 0.995, 0.99.
    adamw_beta2: F,

    /// AdamW backpropagation optimization hyperparameter.
    ///
    /// Weight decay is a regularization technique that adds a penalty term
    /// to the loss function to discourage large weights. A high value
    /// of weight decay means that the optimizer will penalize large weights
    /// more heavily.
    ///
    /// Recommended values: 0.01, 0.001, 0.0001.
    adamw_lambda: F,

    /// Amount of backpropagation timesteps to reach the target learn rate value.
    ///
    /// If set to 0 then no warmup will be applied.
    warmup_duration: u32,

    /// Maximal difference from the target learn rate value.
    ///
    /// Actual learn rate will vary in range `[target - radius, target + radius]`.
    ///
    /// If set to 0 then no cyclic schedule will be applied.
    cycle_radius: F,

    /// Amount of timesteps of backpropagation for going full cicle
    /// of learning rate changing.
    ///
    /// If set to 0 then no cyclic schedule will be applied.
    cycle_period: u32,

    /// Target learn rate of the backpropagation.
    ///
    /// It is different from the actual learn rate because we use cyclic
    /// learn rate update schedule and warmup phase. During warmup learning
    /// rate will slowly increase from `F::EPSILON` to the current value,
    /// and then it will start going lower and higher according to the
    /// cyclic schedule around the target value.
    ///
    /// ```text,ignore
    ///                learn rate
    ///                    ^
    ///                    |        . .         . .
    /// target learn rate  +      .     .     .     .     .
    ///                    |    +         . .         . .
    ///                    |   /
    ///                    |  /
    ///                    | /
    ///                    +-------------------------------> time
    ///                     ^^^^^^ warmup
    /// ```
    learn_rate: F
}

impl<const SIZE: usize, F: Float> Default for Backpropagation<SIZE, F> {
    fn default() -> Self {
        Self {
            timestep: 0,

            adamw_m: vec![F::ZERO; SIZE].into_boxed_slice(),
            adamw_v: vec![F::ZERO; SIZE].into_boxed_slice(),

            // Recommended defaults for AdamW optimizer.
            adamw_beta1: F::from_float(0.9),
            adamw_beta2: F::from_float(0.999),
            adamw_lambda: F::from_float(0.01),

            // Do not do learn rate warmup.
            warmup_duration: 0,

            // Do not do cyclic schedule for the learn rate.
            cycle_radius: F::ZERO,
            cycle_period: 0,

            // Some default learn rate value generated by LLM with this reasoning:
            //
            // - Works well as a max learning rate when using AdamW.
            // - Acts as a balanced starting point for cyclic schedules.
            // - Empirically validated in many setups.
            learn_rate: F::from_float(0.0003)
        }
    }
}

impl<const SIZE: usize, F: Float> Backpropagation<SIZE, F> {
    /// Try to find best backpropagation parameters by slowly increasing
    /// the learning rate and testing results with provided loss function.
    ///
    /// This method will try to predict the most suitable learn rate value
    /// for default AdamW optimizer parameters. It will not use warmup stage
    /// and cyclic update schedule. You should setup them yourself.
    ///
    /// `exploration_loss` should apply given backpropagation policy
    /// to some model and return its loss value.
    ///
    /// Make sure to use the same model with its *initial* state
    /// to compare impact of learn rate change.
    pub fn explore(exploration_loss: impl Fn(&mut Self) -> f64) -> Self {
        const SMOOTH_WINDOW: usize = 10;
        const SMOOTH_WINDOW_FLOAT: f64 = SMOOTH_WINDOW as f64;

        let mut learn_rate = F::EPSILON;
        let learn_rate_step = F::from_float(2.0);

        // Remember last 10 loss values.
        let mut losses = [1.0; SMOOTH_WINDOW];
        let mut avg_loss = 0.0;

        #[allow(clippy::needless_range_loop)]
        // Fill first 10 loss values.
        for i in 0..10 {
            // Create backpropagation policy with default values and set
            // its learn rate to zero.
            let mut backpropagation = Self::default()
                .with_learn_rate(learn_rate);

            // Calculate loss for the current learn rate value.
            losses[i] = exploration_loss(&mut backpropagation) / SMOOTH_WINDOW_FLOAT;

            avg_loss += losses[i];

            // Increase learn rate and update the policy.
            learn_rate *= learn_rate_step;
        }

        // Keep increasing learn rate until we see the spike in loss value.
        while learn_rate.as_f32() < 1.0 {
            // Create backpropagation policy with default values and set
            // its learn rate to zero.
            let mut backpropagation = Self::default()
                .with_learn_rate(learn_rate);

            // Calculate loss for the current learn rate value.
            let mut loss = exploration_loss(&mut backpropagation);

            // Break exploration if found loss is greater than 1.5 of moving average.
            if loss > avg_loss * 1.5 {
                // Use quarter of found learn rate for the final result.
                return Self::default().with_learn_rate(learn_rate * F::HALF * F::HALF);
            }

            // Update moving average.
            loss /= SMOOTH_WINDOW_FLOAT;

            avg_loss -= losses[0];
            avg_loss += loss;

            for j in 0..9 {
                losses[j] = losses[j + 1];
            }

            losses[9] = loss;

            // Increase learn rate and update the policy.
            learn_rate *= learn_rate_step;
        }

        // Return default policy when failed to opimize.
        Self::default()
    }

    #[inline]
    /// Change AdamW beta1 parameter.
    pub fn with_adamw_beta1(mut self, adamw_beta1: F) -> Self {
        self.adamw_beta1 = adamw_beta1;

        self
    }

    #[inline]
    /// Change AdamW beta2 parameter.
    pub fn with_adamw_beta2(mut self, adamw_beta2: F) -> Self {
        self.adamw_beta2 = adamw_beta2;

        self
    }

    #[inline]
    /// Change AdamW lambda parameter.
    pub fn with_adamw_lambda(mut self, adamw_lambda: F) -> Self {
        self.adamw_lambda = adamw_lambda;

        self
    }

    #[inline]
    /// Change target learn rate of the backpropagation.
    pub fn with_learn_rate(mut self, learn_rate: F) -> Self {
        self.learn_rate = learn_rate;

        self
    }

    #[inline]
    /// Change warmup stage duration in backpropagation timesteps.
    pub fn with_warmup_duration(mut self, warmup_duration: u32) -> Self {
        self.warmup_duration = warmup_duration;

        self
    }

    #[inline]
    /// Change cycle radius of the cyclic learn rate schedule.
    pub fn with_cycle_radius(mut self, cycle_radius: F) -> Self {
        self.cycle_radius = cycle_radius;

        self
    }

    #[inline]
    /// Change cycle period of the cyclic learn rate schedule in backpropagation timesteps.
    pub fn with_cycle_period(mut self, cycle_period: u32) -> Self {
        self.cycle_period = cycle_period;

        self
    }

    #[inline]
    /// Call given callback with a snapshot of the backpropagation policy
    /// with timestep increased by one.
    pub fn timestep<T>(&mut self, mut callback: impl FnMut(BackpropagationSnapshot<'_, SIZE, F>) -> T) -> T {
        self.timestep += 1;

        callback(BackpropagationSnapshot(self))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
/// Wrapper of the backpropagation policy to enforce methods execution
/// within the same timestep for all neurons and layers within the network.
pub struct BackpropagationSnapshot<'policy, const SIZE: usize, F: Float>(&'policy mut Backpropagation<SIZE, F>);

impl<const SIZE: usize, F: Float> BackpropagationSnapshot<'_, SIZE, F> {
    /// Call given callback with a slice of current backpropagation policy struct
    /// and update current one after the callback execution.
    ///
    /// Note that `WINDOW_SIZE` is allowed to exceed actually stored gradients amount
    /// (size of the backpropagation policy), but new gradients will have default
    /// values while timestep won't, which will directly affect training quality.
    pub fn window<const WINDOW_SIZE: usize, T>(&mut self, offset: usize, mut callback: impl FnMut(BackpropagationSnapshot<WINDOW_SIZE, F>) -> T) -> T {
        let mut windowed = Backpropagation {
            timestep: self.0.timestep,

            adamw_m: (0..WINDOW_SIZE).map(|i| {
                self.0.adamw_m.get(offset + i)
                    .copied()
                    .unwrap_or(F::ZERO)
            }).collect(),

            adamw_v: (0..WINDOW_SIZE).map(|i| {
                self.0.adamw_v.get(offset + i)
                    .copied()
                    .unwrap_or(F::ZERO)
            }).collect(),

            adamw_beta1: self.0.adamw_beta1,
            adamw_beta2: self.0.adamw_beta2,
            adamw_lambda: self.0.adamw_lambda,

            warmup_duration: self.0.warmup_duration,

            cycle_radius: self.0.cycle_radius,
            cycle_period: self.0.cycle_period,

            learn_rate: self.0.learn_rate
        };

        let output = callback(BackpropagationSnapshot(&mut windowed));

        // Clone updated gradients from the windowed policy.
        if offset < SIZE {
            let available = std::cmp::min(SIZE - offset, WINDOW_SIZE);

            self.0.adamw_m[offset..offset + available].copy_from_slice(&windowed.adamw_m[..available]);
            self.0.adamw_v[offset..offset + available].copy_from_slice(&windowed.adamw_v[..available]);
        }

        output
    }

    /// Backpropagate given values using calculated gradients.
    ///
    /// It's expected that values and gradients have `SIZE` values.
    pub fn backpropagate(&mut self, values: &mut Box<[F; SIZE]>, gradients: &Box<[F; SIZE]>) {
        // Learn rate warmup.
        let learn_rate = if self.0.timestep < self.0.warmup_duration {
            self.0.learn_rate * F::from_float(self.0.timestep as f32 / self.0.warmup_duration as f32)
        }

        // Either period or radius of the cycle are zero, which mean
        // we can't apply cyclic schedule to the learn rate.
        else if self.0.cycle_period == 0 || self.0.cycle_radius == F::ZERO {
            self.0.learn_rate
        }

        // Cyclic schedule.
        else {
            // Amount of passed timesteps within the cycle.
            let cycle_steps = (self.0.timestep - self.0.warmup_duration) % self.0.cycle_period;

            // Percent of cycle finishing.
            let cycle_value = cycle_steps as f32 / self.0.cycle_period as f32;

            // Calculate current sine value, scale it by the radius and add it to the learn rate.
            self.0.learn_rate + self.0.cycle_radius * F::from_float((2.0 * std::f32::consts::PI * cycle_value).sin())
        };

        // Calculate AdamW beta powers.
        let adamw_inv_beta1_t = F::ONE - self.0.adamw_beta1.powi(self.0.timestep as i32);
        let adamw_inv_beta2_t = F::ONE - self.0.adamw_beta2.powi(self.0.timestep as i32);

        // Update given values.
        for i in 0..SIZE {
            // Skip weights updating when gradient is 0.
            if gradients[i] != F::ZERO {
                // Update AdamW moving averages.
                self.0.adamw_m[i] = self.0.adamw_beta1 * self.0.adamw_m[i] + (F::ONE - self.0.adamw_beta1) * gradients[i];
                self.0.adamw_v[i] = self.0.adamw_beta2 * self.0.adamw_v[i] + (F::ONE - self.0.adamw_beta2) * gradients[i].powi(2);

                // Calculate their weighted values.
                let adamw_weighted_m = self.0.adamw_m[i] / adamw_inv_beta1_t;
                let adamw_weighted_v = self.0.adamw_v[i] / adamw_inv_beta2_t;

                // Update value using gradient and calculated AdamW values.
                values[i] -= learn_rate * adamw_weighted_m / (adamw_weighted_v.sqrt() + F::EPSILON) + learn_rate * self.0.adamw_lambda * values[i];
            }
        }
    }
}
