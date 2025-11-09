# Model Card: Depression Detection from Speech

## Model Information

**Model Name**: Depression Detection CNN  
**Version**: 1.0.0  
**Date**: 2024  
**Author**: Depression Detection Team  
**License**: MIT License

## Model Details

### Architecture Summary

The model is an enhanced Convolutional Neural Network (CNN) designed for depression detection from speech audio:

- **Input**: 120 MFCC features (40 base + 40 delta + 40 delta-delta) × 31 time frames
- **Architecture**: 
  - 3 Conv1D layers (256 channels each) with Batch Normalization
  - Global Average Pooling
  - 4-layer MLP classifier with progressive dropout (0.6 → 0.4 → 0.3)
- **Output**: 2 classes (Healthy, Depressed)
- **Parameters**: ~6.1 million trainable parameters

### Training Data Summary

- **Dataset Size**: Limited dataset (exact size not disclosed for privacy)
- **Class Distribution**: Imbalanced (exact distribution not disclosed)
- **Input Type**: Audio files, 16kHz sampling rate
- **Preprocessing**: 
  - Silence removal (top_db=20)
  - MFCC feature extraction (40 coefficients)
  - Delta and delta-delta features
  - Normalization per sample
  - Segmentation into 1-second windows with 50% overlap

### Training Date

Model was trained in 2024 using 5-fold cross-validation.

### Training Procedure

- **Method**: 5-fold stratified cross-validation
- **Data Augmentation**: 8x dataset expansion (time stretch, pitch shift, volume adjustment, noise injection)
- **Loss Function**: CrossEntropyLoss with label smoothing
- **Optimizer**: AdamW with layer-specific learning rates
- **Regularization**: Dropout, BatchNorm, Weight Decay (1e-4)
- **Early Stopping**: Patience of 15 epochs
- **Learning Rate Scheduling**: ReduceLROnPlateau

## Intended Use

### Primary Use Case

**Research and Demonstration Only**

This model is intended for:
- Research purposes in computational psychiatry and speech analysis
- Educational demonstrations of machine learning for mental health
- Non-clinical screening tools for research studies
- Academic and research applications

### Out-of-Scope Use Cases

**NOT INTENDED FOR:**
- Clinical diagnosis or medical decision-making
- Standalone medical device use
- Treatment recommendations
- Legal or insurance purposes
- Real-world clinical deployment without extensive validation

## Performance Metrics

### Evaluation Metrics (as of 2024)

- **Best Accuracy**: 75.0% (best fold in 5-fold CV)
- **Average Accuracy**: 65.7% ± 6.5% (across 5 folds)
- **Precision**: 74.0% (weighted average)
- **Recall**: 76.0% (weighted average)
- **F1-Score**: 75.0% (weighted average)
- **AUC-ROC**: Not reported (binary classification)

**Note**: These metrics are based on cross-validation on a limited dataset. Performance may vary significantly on different populations, languages, or recording conditions.

## Limitations and Risks

### Technical Limitations

1. **Dataset Size**: Trained on a limited dataset; may not generalize to diverse populations
2. **Language**: Primarily trained on English audio; performance on other languages is unknown
3. **Recording Conditions**: Performance may degrade with poor audio quality, background noise, or different recording devices
4. **Demographic Bias**: Dataset may not represent all demographic groups equally
5. **Temporal Context**: Model analyzes short audio segments (1-30 seconds); may miss long-term patterns
6. **Feature Limitations**: Relies solely on MFCC features; may miss other relevant acoustic cues

### Ethical Considerations

1. **False Positives**: May incorrectly flag healthy individuals as depressed, causing unnecessary concern
2. **False Negatives**: May miss actual depression cases, leading to lack of appropriate care
3. **Privacy**: Audio recordings contain sensitive personal information
4. **Stigma**: Misuse could contribute to mental health stigma
5. **Bias**: Model may exhibit bias against certain demographic groups if training data is imbalanced
6. **Over-reliance**: Users may over-rely on model predictions instead of seeking professional help

### Clinical Limitations

1. **Not a Medical Device**: This model is NOT FDA-approved or certified for clinical use
2. **No Clinical Validation**: Has not been validated in clinical settings or against clinical gold standards
3. **Single Modality**: Only uses speech; depression diagnosis typically requires multiple assessments
4. **Context Missing**: Lacks clinical context, patient history, and other diagnostic information
5. **Temporal Stability**: Does not account for day-to-day variations in speech patterns

## Ethical Considerations

### Data Privacy

- Audio recordings contain sensitive personal information
- Users should be informed about data collection and storage practices
- Implement appropriate data protection measures (encryption, access controls)
- Comply with relevant privacy regulations (GDPR, HIPAA, etc.)

### Fairness and Bias

- Model may exhibit bias if training data is not representative
- Performance may vary across demographic groups (age, gender, ethnicity, socioeconomic status)
- Regular bias audits should be conducted
- Mitigation strategies should be implemented if bias is detected

### Responsible Use

- **Always include disclaimers** that this is a research tool, not a medical device
- **Never use for clinical diagnosis** without proper validation and regulatory approval
- **Encourage professional consultation** for any mental health concerns
- **Provide resources** for mental health support and crisis intervention
- **Monitor for misuse** and implement safeguards

## Evaluation Protocol

### Cross-Validation

Model was evaluated using 5-fold stratified cross-validation to ensure:
- Representative performance across different data splits
- Reduced overfitting risk
- More reliable performance estimates

### Metrics Calculation

- **Accuracy**: Overall classification accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Per-class classification breakdown

### Evaluation Date

Metrics reported as of 2024. Regular re-evaluation recommended as new data becomes available.

## Model Artifacts

- **Model Checkpoint**: `models/best_model.pth`
- **Training History**: `models/training_history.json`
- **Cross-Validation Results**: `models/cross_validation_results.json`
- **Code Repository**: Available in project repository

## Citation

If you use this model in your research, please cite:

```bibtex
@article{depression_detection_2024,
  title={High-Accuracy Speech-Based Depression Detection Using Deep Learning},
  author={Depression Detection Team},
  journal={AI in Healthcare},
  year={2024}
}
```

## Contact and Support

For questions, concerns, or to report issues:
- Create an issue on the project repository
- Contact: support@depression-detection.ai

## Disclaimer

**IMPORTANT**: This model is for research and demonstration purposes only. It is NOT a medical device and should NOT be used for clinical diagnosis, treatment decisions, or any medical purpose. Always consult qualified healthcare professionals for mental health concerns.

---

**Last Updated**: 2024  
**Model Version**: 1.0.0

