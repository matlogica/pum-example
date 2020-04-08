#include <iostream>
#include <aadc/ibool.h>
#include <aadc/aadc.h>
#include <random>
#include "examples.h"
#include <aadc/AADCMatrixInterface.h>
#include <thread>

using namespace aadc;
typedef double Time;

template<class vtype>
class BankRate {
public: 
    BankRate (vtype _rate) : rate(_rate) {}
    vtype operator () (const Time& t) const {return rate;}
    ~BankRate() {}
public:
    vtype rate;
};

template<class vtype>
class AssetVolatility {
public: 
    AssetVolatility (vtype _vol) : vol(_vol) {}
    vtype operator () (const vtype asset, const Time& t) const {return vol;}
    ~AssetVolatility() {}
public:
    vtype vol;
};

template<class vtype>
vtype simulateAssetOneStep(
    const vtype current_value
    , Time current_t
    , Time next_t
    , const BankRate<vtype>& rate
    , const AssetVolatility<vtype>& vol_obj
    , const vtype& random_sample
) {
    double dt = (next_t-current_t);
    vtype vol=vol_obj(current_value, current_t);
    vtype next_value = current_value * (
        1 + (-vol*vol / 2+ rate(current_t))*dt  
        +  vol * std::sqrt(dt) * random_sample
    );
    return next_value;
}

template<class vtype>
vtype onePathPricing (
    vtype asset
    , double strike
    , const std::vector<Time>& t
    , const BankRate<vtype>& rate_obj
    , const AssetVolatility<vtype>& vol_obj
    , const std::vector<vtype>& random_samples
) {
    for (int t_i = 0; t_i < t.size()-1; ++t_i) {
        asset = simulateAssetOneStep(asset, t[t_i], t[t_i+1], rate_obj, vol_obj, random_samples[t_i]);
    }
    return std::max(asset - strike, 0.);
}

inline double compwiseSum(const std::vector<double>& vec) 
{
    double sum(0.);
    for(int i=0; i<vec.size(); i++) sum+=vec[i];
    return sum;
}

void example_option_pricing()
{
    int num_threads=4;
    typedef __m256d mmType;  // __mm256d and __mm512d are supported
    
    double init_vol(0.005), init_asset(100.), init_rate(0.3);

    std::vector<Time> t(100 , 0.); 
    int num_time_steps=t.size();
    for (int i=1; i<num_time_steps; i++) t[i]=i*0.01;

    double strike=95.;
    int num_mc_paths=1600;
    int AVXsize = sizeof(mmType) / sizeof(double);
    int paths_per_thread = num_mc_paths / (AVXsize*num_threads);    
    //assert(num_mc_paths % (num_threads *AADC::getNumAvxElements<mmType>()) == 0);    
    
    aadc::AADCFunctions<mmType> aad_funcs;

    std::vector<idouble> aad_random_samples(num_time_steps, 0.);
    idouble aad_rate(init_rate), aad_vol(init_vol), aad_asset(init_asset);
    aadc::VectorArg random_arg;

    aad_funcs.startRecording();
        // Mark vector of random variables as input only. No adjoints for them
        markVectorAsInput(random_arg, aad_random_samples, false);          
        
        aadc::AADCArgument rate_arg(aad_rate.markAsInput());
        aadc::AADCArgument asset_arg(aad_asset.markAsInput());
        aadc::AADCArgument vol_arg(aad_vol.markAsInput());
        
        BankRate<idouble> rate_obj(aad_rate);
        AssetVolatility<idouble> vol_obj(aad_vol);

        idouble payoff= onePathPricing(aad_asset, strike, t, rate_obj, vol_obj, aad_random_samples);
        aadc::AADCResult payoff_arg(payoff.markAsOutput());
    aad_funcs.stopRecording();

    std::vector<double> total_prices(num_threads, 0.)
        , deltas(num_threads, 0.)
        , vegas(num_threads, 0.)
        , rhos(num_threads, 0.)
    ;
    
    auto threadWorker = [&] (
        double& total_price
        , double& vega 
        , double& delta 
        , double& rho 
    ) {
        std::mt19937_64 gen;  
        std::normal_distribution<> normal_distrib(0.0, 1.0);            
        std::shared_ptr<aadc::AADCWorkSpace<mmType> > ws(aad_funcs.createWorkSpace());

        mmType mm_total_price(mmSetConst<mmType>(0))
            , mm_vega(mmSetConst<mmType>(0))
            , mm_delta(mmSetConst<mmType>(0))
            , mm_rho(mmSetConst<mmType>(0))
        ;

        aadc::AVXVector<mmType> randoms(num_time_steps);

        for (int mc_i = 0; mc_i < paths_per_thread; ++mc_i) {
            for (int j=0; j<num_time_steps; j++) {
                for (int c=0; c<AVXsize; c++) toArray(randoms[j])[c]=normal_distrib(gen);
            }
            setAVXVector(*ws, random_arg, randoms);
            ws->val(rate_arg) = mmSetConst<mmType>(init_rate);
            ws->val(asset_arg) = mmSetConst<mmType>(init_asset);
            ws->val(vol_arg) = mmSetConst<mmType>(init_vol);
                
            aad_funcs.forward(*ws);
            mm_total_price=mmAdd(ws->val(payoff_arg), mm_total_price);
            //std::cout << ws->val(payoff_arg)[1] << " Val\n";
            ws->setDiff(payoff_arg, 1.0);
            aad_funcs.reverse(*ws);        

            mm_vega=mmAdd(ws->diff(vol_arg), mm_vega);
            mm_delta=mmAdd(ws->diff(asset_arg), mm_delta);
            mm_rho=mmAdd(ws->diff(rate_arg), mm_rho);
        }    
        
        total_price=aadc::mmSum(mm_total_price)/num_mc_paths;
        rho=aadc::mmSum(mm_rho)/num_mc_paths;
        delta=aadc::mmSum(mm_delta)/num_mc_paths;
        vega=aadc::mmSum(mm_vega)/num_mc_paths;
    };
    
    std::vector<std::unique_ptr<std::thread>> threads;
    for(int i=0; i< num_threads; i++) {
        threads.push_back(
            std::unique_ptr<std::thread>(
                new std::thread(
                    threadWorker
                    , std::ref(total_prices[i])
                    , std::ref(vegas[i])
                    , std::ref(deltas[i])
                    , std::ref(rhos[i])                                        
                )
            )
        );
    }
    for(auto&& t: threads) t->join();
            
    std::cout << "Price " << compwiseSum(total_prices) << "\n";
    std::cout << "Delta " << compwiseSum(deltas) << "\n";
    std::cout << "Vega " << compwiseSum(vegas) << "\n";
    std::cout << "Rho " << compwiseSum(rhos) << "\n";
}