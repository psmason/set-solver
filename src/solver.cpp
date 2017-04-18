#include <solver.h>

#include <map>
#include <set>
#include <iostream>

namespace setsolver {

  namespace {
    using namespace std;
     
    using Colors         = map<Color, size_t>;
    using SymbolBuckets  = map<Symbol, Colors>;
    using ShadingBuckets = map<Shading, SymbolBuckets>;
    using NumberBuckets  = map<size_t, ShadingBuckets>;

    NumberBuckets organize(const FeatureSet& featureSet) {
      NumberBuckets numberBuckets;
      for (size_t i=0; i<featureSet.size(); ++i) {
        const auto& feature = featureSet[i];
        numberBuckets[feature.number]
          [feature.shading]
          [feature.symbol]
          [feature.color] = i;
      }
      return numberBuckets;
    }

    bool getSatisfyingIndex(const NumberBuckets& numberBuckets,
                            const CardFeatures& satisfyingFeatures,
                            size_t& index) {
      const auto shadingBucketsItr = numberBuckets.find(satisfyingFeatures.number);
      if (numberBuckets.end() == shadingBucketsItr) {
        return false;
      }
      const auto& shadingBuckets = shadingBucketsItr->second;
      
      const auto symbolBucketsItr = shadingBuckets.find(satisfyingFeatures.shading);
      if (shadingBuckets.end() == symbolBucketsItr) {
        return false;
      }
      const auto& symbolBuckets = symbolBucketsItr->second;

      const auto colorBucketsItr = symbolBuckets.find(satisfyingFeatures.symbol);
      if (symbolBuckets.end() == colorBucketsItr) {
        return false;
      }
      const auto& colorBuckets = colorBucketsItr->second;

      const auto indexItr = colorBuckets.find(satisfyingFeatures.color);
      if (colorBuckets.end() == indexItr) {
        return false;
      }

      index = indexItr->second;
      return true;
    }

    template <typename T>
    T getSatisfyingFeature(const T& f1,
                           const T& f2,
                           set<T> featureSet) {
      if (f1 == f2) {
        return f1;
      }
      else {
        featureSet.erase(f1);
        featureSet.erase(f2);
        assert(1 == featureSet.size());
        return *featureSet.begin();
      }
    }

    CardFeatures getSatisfyingCard(const CardFeatures& card1,
                                   const CardFeatures& card2) {
      // for each card feature, all three cards must be equivalent
      // or all different.
      CardFeatures feature;

      feature.color = getSatisfyingFeature(card1.color,
                                           card2.color,
                                           set<Color>{
                                             Color::GREEN,
                                             Color::RED,
                                             Color::PURPLE
                                           });

      feature.symbol = getSatisfyingFeature(card1.symbol,
                                            card2.symbol,
                                            set<Symbol>{
                                              Symbol::DIAMOND,
                                              Symbol::SQUIGGLE,
                                              Symbol::OVAL
                                            });
      
      feature.shading = getSatisfyingFeature(card1.shading,
                                             card2.shading,
                                             set<Shading>{
                                               Shading::SOLID,
                                               Shading::STRIPED,
                                               Shading::OPEN
                                             });

      feature.number = getSatisfyingFeature(card1.number,
                                            card2.number,
                                            set<size_t>{
                                              1,
                                              2,
                                              3,
                                            });
      
      return feature;
    }
    
  } // close anonymous
  
  Matches findMatches(const FeatureSet& featureSet)
  {
    const auto numberBuckets = organize(featureSet);

    // basic property of set is that given any two cards,
    // a third will satify the match condition.
    // so let's iterate over all pairs and see if the third
    // card is available for the match.

    Matches matches;
    for (size_t i=0; i<featureSet.size(); ++i) {
      for (size_t j=i+1; j<featureSet.size(); ++j) {
        const auto satisfyingCard = getSatisfyingCard(featureSet[i],
                                                      featureSet[j]);
        size_t satisfyingIndex;
        if (getSatisfyingIndex(numberBuckets,
                               satisfyingCard,
                               satisfyingIndex)) {
          Match candidate = {i, j, satisfyingIndex};
          // to avoid duplicates, only add matches which are
          // in sorted order.
          if (std::is_sorted(candidate.begin(), candidate.end())) {
            // in case of featur extraction errors, check if the indexes are
            // unique. 
            if (candidate.end() == std::unique(candidate.begin(), candidate.end())) {
              matches.push_back(candidate);
            }
          }
        }
      }
    }    
    return matches;
  }
  
}
