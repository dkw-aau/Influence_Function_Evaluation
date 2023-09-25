distributions = torch.zeros(len(Y_train))
for i in range(200):
    test_point_scores = module.influences(train_idxs=train_idxs, test_idxs=[i])
    
    distributions+=test_point_scores.sort().values

# Plot the average distribution
plt.hist(distributions, bins=130)
plt.xticks(fontsize=14)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.yticks(fontsize=13)
plt.xlabel("Influence Score", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.savefig('inf_cnn_sortsum.eps', format='eps')
plt.show()




 def draw(test_indx, train_idxs): 
    # ===========
    # Plot image
    # =========
    new_line = '\n'
    fig, axs = plt.subplots(3, 3, figsize=(11, 11))
    axs[0, 0].imshow(captioned_image(clf, embeds, 'test', test_indx))
    axs[0, 0].axis('off')
    axs[0, 0].text(0.5, -0.15, f"Prediction: {clf(X_test[test_indx].unsqueeze(0)).round().item()} {new_line} Actual Label: {Y_test[test_indx].item()}", size=14, ha="center", transform=axs[0, 0].transAxes)
    axs[1, 0].text(-0.1, 0, 'Influential Samples',size=14, rotation=90, va='center', ha='right', transform=axs[1, 0].transAxes)
    axs[0, 0].set_title('Test Prediction',size=14)
    
    for i, j in enumerate((-train_idxs).argsort()[:6]):
        if i<3:
            axs[1, i].imshow(captioned_image(clf, embeds, 'train', j)) 
            axs[1, i].text(0.5, -0.1, f"Label: {Y_train[j.item()].item()}", size=14, ha="center", transform=axs[1, i].transAxes)
            axs[1, i].axis('off')
            axs[0, i].axis('off')
        else:
            axs[2, i-3].imshow(captioned_image(clf, embeds, 'train', j)) 
            axs[2, i-3].text(0.5, -0.1, f"Label: {Y_train[j.item()].item()}", size=14, ha="center", transform=axs[2, i-3].transAxes)
            axs[2, i-3].axis('off')
            axs[0, i-3].axis('off')
        
#     for i, j in enumerate(train_idxs.argsort()[:6]):
#         axs[2, i].imshow(captioned_image(clf, embeds, 'train', j))
#         axs[2, i].axis('off')
#         axs[2, i].text(0.5, -0.15, f"Influence: {train_idxs[j]:.8f} {new_line} Label: {Y_train[j.item()].item()}", size=10, ha="center", transform=axs[2, i].transAxes)

    plt.savefig(f'inf_{test_indx}.eps', format='eps', bbox_inches="tight")

    plt.show() 
    
draw(test_idx,np.array(df.Influence.tolist()))    

 def drawX(test_indx, train_idxs): 
    # ===========
    # Plot image
    # =========
    new_line = '\n'
    fig, axs = plt.subplots(3, 6, figsize=(11, 11))
    axs[0, 0].imshow(captioned_image(clf, embeds, 'test', test_indx))
    axs[0, 0].axis('off')
    axs[0, 0].text(0.5, -0.15, f"Prediction: {clf(X_test[test_indx].unsqueeze(0)).round().item()} {new_line} Actual Label: {Y_test[test_indx].item()}", size=12, ha="center", transform=axs[0, 0].transAxes)

    for i, j in enumerate((-train_idxs).argsort()[:6]):
     
        axs[1, i].imshow(captioned_image(clf, embeds, 'train', j)) 
        axs[1, i].text(0.5, -0.1, f"Label: {Y_train[j.item()].item()}", size=12, ha="center", transform=axs[1, i].transAxes)
        axs[1, i].axis('off')
        axs[0, i].axis('off')

        
    for i, j in enumerate(train_idxs.argsort()[:6]):
        axs[2, i].imshow(captioned_image(clf, embeds, 'train', j))
        axs[2, i].axis('off')
        axs[2, i].text(0.5, -0.15, f"Influence: {train_idxs[j]:.8f} {new_line} Label: {Y_train[j.item()].item()}", size=10, ha="center", transform=axs[2, i].transAxes)

    axs[0, 0].set_title('Test Prediction')
    axs[1, 1].set_title('Influential Samples')
#     plt.savefig(f'inf_{test_indx}.eps', format='eps', bbox_inches="tight")

    axs[2, 0].set_title('Harmfull Images')
    plt.show() 
    
drawX(test_idx,np.array(df.Influence.tolist()))



errors=[20, 266, 300, 368, 428, 458, 484, 507]
# plot misclassified fishes due to human
test_idx0 = 20
test_idx1 = 428
fig, axs = plt.subplots(1, 2, figsize=(11, 5))
axs[0].imshow(captioned_image(clf, embeds, 'test', test_idx0))
axs[1].imshow(captioned_image(clf, embeds, 'test', test_idx1))
axs[0].axis('off')
axs[1].axis('off')
axs[0].text(0.5, -0.15, f"Prediction: {clf(X_test[20].unsqueeze(0)).round().item()} \n Actual Label: {Y_test[20].item()}", size=18, ha="center", transform=axs[0].transAxes)
axs[1].text(0.5, -0.15, f"Prediction: {clf(X_test[428].unsqueeze(0)).round().item()} \n Actual Label: {Y_test[428].item()}", size=18, ha="center", transform=axs[1].transAxes)
plt.savefig('errors.eps', format='eps', bbox_inches="tight")
plt.show()

'''
Wrongly classified
1.0 0.0 20  market fish
0.0 1.0 266 cut dogs
1.0 0.0 300 people and fish
1.0 0.0 368 fish
1.0 0.0 428 market fish
1.0 0.0 458 ambiguous
1.0 0.0 484 nothing
0.0 1.0 507
462 from pang  [20, 266, 300, 368, 428, 458, 484, 507]

Ambiguous
69 crocodile
76
140  presentation clarif
185 dog water
286 eye
305 both
314 not clear
362,                       455 dog water


both as fish = 650, 1616, 1190


# # # fig, axs = plt.subplots(9, 5, figsize=(14, 14))
# # # x=550
# # # y=555
# # # for k in range(x,y):
# # #     axs[0,k-x].imshow(captioned_image(clf, embeds, 'test', k))
# # #     axs[0, k-x].axis('off')
# # #     axs[1,k-x].imshow(captioned_image(clf, embeds, 'test', k+5))
# # #     axs[1, k-x].axis('off')
# # #     axs[2,k-x].imshow(captioned_image(clf, embeds, 'test', k+10))
# # #     axs[2, k-x].axis('off')
# # #     axs[3,k-x].imshow(captioned_image(clf, embeds, 'test', k+15))
# # #     axs[3, k-x].axis('off')
# # #     axs[4,k-x].imshow(captioned_image(clf, embeds, 'test', k+20))
# # #     axs[4, k-x].axis('off')
# # #     axs[5,k-x].imshow(captioned_image(clf, embeds, 'test', k+25))
# # #     axs[5, k-x].axis('off')
# # #     axs[6,k-x].imshow(captioned_image(clf, embeds, 'test', k+30))
# # #     axs[6, k-x].axis('off')
# # #     axs[7,k-x].imshow(captioned_image(clf, embeds, 'test', k+35))
# # #     axs[7, k-x].axis('off')
# # #     axs[8,k-x].imshow(captioned_image(clf, embeds, 'test', k+40))
# # #     axs[8, k-x].axis('off')



image = captioned_image(clf, embeds, 'train', 1141)
plt.imshow(image)
plt.axis('off')
plt.savefig('image1141.svg', format='svg')
plt.show()


def print_result(test_idx, selected_indices_neg_ol=None, selected_indices_neg_sl=None,
                 selected_indices_pos_sl=None, selected_indices_pos_ol=None,
                 df_neg_ol=None, df_neg_sl=None,
                 df_pos_sl=None, df_pos_ol=None, intent = None): 
    
    new_line = '\n'
    fig, axs = plt.subplots(5, 3, figsize=(11, 15))
    axs[0, 0].imshow(captioned_image(clf, embeds, 'test', test_idx))
    axs[0, 0].axis('off')
    axs[0, 0].text(0.5, -0.18, f"Prediction: {clf(X_test[test_idx].unsqueeze(0)).round().item()}{new_line} Actual Label: {Y_test[test_idx].item()}", size=12, ha="center", transform=axs[0, 0].transAxes)
    axs[0, 0].set_title('Test Prediction')
    if intent=='interpret': 
        axs[1, 0].text(-0.1, 0.5, 'Support by Relevance',size=14, rotation=90, va='center', ha='right', transform=axs[1, 0].transAxes)
        axs[2, 0].text(-0.1, 0.5, 'Support by Contrast',size=14, rotation=90, va='center', ha='right', transform=axs[2, 0].transAxes)
    else:
        axs[1, 0].text(-0.1, 0.5, 'Support by Relevance',size=14, rotation=90, va='center', ha='right', transform=axs[1, 0].transAxes)
        axs[2, 0].text(-0.1, 0.5, 'Support by Contrast',size=14, rotation=90, va='center', ha='right', transform=axs[2, 0].transAxes)
        axs[3, 0].text(-0.1, 0.5, 'Oppose by Relevance',size=14, rotation=90, va='center', ha='right', transform=axs[3, 0].transAxes)
        axs[4, 0].text(-0.1, 0.5, 'Oppose by Contrast',size=14, rotation=90, va='center', ha='right', transform=axs[4, 0].transAxes)
 
    
    
    for i, j in enumerate(selected_indices_pos_sl[:3]):
        axs[3, i].imshow(captioned_image(clf, embeds, 'train', df_pos_sl.Influence.index[j])) 
        axs[3, i].axis('off')
        axs[0, i].axis('off')
        axs[3, i].text(0.5, -0.15, f"Label: {df_pos_sl.Y_train.tolist()[j]}", size=14, ha="center", transform=axs[3, i].transAxes)
    #Influence: {df_pos_sl.Influence.tolist()[j]:.8f}{new_line} 
    for i, j in enumerate(selected_indices_pos_sl[3:]):
        axs[4, i].imshow(captioned_image(clf, embeds, 'train', df_pos_sl.Influence.index[j])) 
        axs[4, i].axis('off')
        axs[4, i].text(0.5, -0.15, f"Label: {df_pos_sl.Y_train.tolist()[j]}", size=14, ha="center", transform=axs[4, i].transAxes)

    
    
#     for i, j in enumerate(selected_indices_pos_ol):
        
#         axs[4, i].imshow(captioned_image(clf, embeds, 'train', df_pos_ol.Influence.index[j]))
#         axs[4, i].axis('off')
#         axs[4, i].text(0.5, -0.15, f"Label: {df_pos_ol.Y_train.tolist()[j]}", size=14, ha="center", transform=axs[4, i].transAxes)
    for i, j in enumerate(selected_indices_neg_ol):
        
        axs[1, i].imshow(captioned_image(clf, embeds, 'train', df_neg_ol.Influence.index[j])) 
        axs[1, i].axis('off')
        axs[1, i].text(0.5, -0.15, f"Label: {df_neg_ol.Y_train.tolist()[j]}", size=14, ha="center", transform=axs[1, i].transAxes)
    
    for i, j in enumerate(selected_indices_neg_sl):
        axs[2, i].imshow(captioned_image(clf, embeds, 'train', df_neg_sl.Influence.index[j])) 
        axs[2, i].axis('off')
        axs[2, i].text(0.5, -0.15, f"Label: {df_neg_sl.Y_train.tolist()[j]}", size=14, ha="center", transform=axs[2, i].transAxes)
    
    
    
#     plt.savefig(f'ibd_{test_idx}_{intent}.eps', format='eps', bbox_inches="tight")

    plt.show() 
    
    
    
print_result(test_idx, selected_indices_neg_ol=selected_indices_neg_ol, selected_indices_neg_sl=selected_indices_neg_sl,
                 selected_indices_pos_sl=selected_indices_pos_sl, selected_indices_pos_ol=selected_indices_pos_ol,
                 df_neg_ol=df_neg_ol, df_neg_sl=df_neg_sl,
                 df_pos_sl=df_pos_sl, df_pos_ol=df_pos_ol, intent = 'interpret')