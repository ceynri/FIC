<template>
  <div class="container">
    <ImageCard class="preview_panel" :src="image.dataUrl" :name="image.name">
      <div>{{ sizeFormat(image.size) }}</div>
    </ImageCard>
    <div class="setting_panel card shadow">
      <div class="setting_item">
        <div class="setting_name">Feature model</div>
        <div class="setting_value_wrapper">
          <label class="setting_value">
            <input type="radio" v-model="value.featureModel" value="facenet" />
            FaceNet
          </label>
          <label class="setting_value">
            <input type="radio" v-model="value.featureModel" value="gan" />
            GAN
          </label>
        </div>
      </div>
      <div class="setting_item">
        <div class="setting_name">Quality level</div>
        <div class="setting_value_wrapper">
          <label class="setting_value">
            <input type="radio" v-model="value.qualityLevel" value="low" />
            Low
          </label>
          <label class="setting_value">
            <input type="radio" v-model="value.qualityLevel" value="medium" />
            Medium
          </label>
          <label class="setting_value">
            <input type="radio" v-model="value.qualityLevel" value="high" />
            High
          </label>
        </div>
      </div>
      <button class="next_btn clickable" @click="$emit('next')">
        <div class="text">Process it</div>
        <IconBase width="20px" height="20px" icon-name="next">
          <RightArrowIcon />
        </IconBase>
      </button>
    </div>
  </div>
</template>

<script>
import ImageCard from '@/components/ImageCard.vue';
import RightArrowIcon from '@/components/icons/RightArrowIcon.vue';

export default {
  props: {
    image: {
      type: Object,
      required: true,
    },
    value: {
      type: Object,
      default: () => ({
        featureModel: 'facenet',
        qualityLevel: 'medium',
      }),
    },
  },
  data() {
    return {};
  },
  components: {
    ImageCard,
    RightArrowIcon,
  },
};
</script>

<style lang="scss" scoped>
.container {
  margin: 30px 0;
  display: flex;
  align-items: flex-start;

  .preview_panel {
    width: 360px;
    margin-right: 40px;
    overflow: hidden;
  }

  .setting_panel {
    flex: 1;
    padding: 60px 60px 80px;

    display: flex;
    flex-direction: column;

    position: relative;

    .setting_item {
      flex: 1;
      margin-bottom: 35px;

      .setting_name {
        font-size: 20px;
        margin-bottom: 20px;
      }

      .setting_value_wrapper {
        font-size: 16px;
        display: flex;

        .setting_value {
          margin-right: 80px;
          display: flex;
          align-items: center;

          input[type='radio'] {
            width: 16px;
            height: 16px;
            margin: 8px;
          }
        }
      }
    }

    .next_btn {
      position: absolute;
      left: 35px;
      // right: 40px;
      bottom: 40px;

      display: flex;
      align-items: center;

      color: var(--primary);
      border-radius: 100px;
      padding: 15px 20px;

      transition: box-shadow var(--duration);

      &:hover {
        box-shadow: 2px 4px 32px -4px var(--shadow);
      }

      .text {
        margin: 0 6px -0.1em;
        font-size: 20px;
      }
    }
  }
}
</style>
