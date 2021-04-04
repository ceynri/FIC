<template>
  <div class="container">
    <ImageCard class="preview_panel" :src="image.dataUrl" :name="image.name">
      <div>{{ sizeFormat(image.size) }}</div>
    </ImageCard>
    <div class="setting_panel card shadow">
      <div class="setting_item">ratio</div>
      <div class="setting_item">quality</div>
      <div class="setting_item">xxx...</div>
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
    width: 300px;
    margin-right: 40px;
    overflow: hidden;
  }

  .setting_panel {
    flex: 1;
    padding: 40px 40px 60px;

    display: flex;
    flex-direction: column;
    // justify-content: space-evenly;

    position: relative;

    .setting_item {
      font-size: 20px;
      margin-bottom: 40px;
    }
    .next_btn {
      position: absolute;
      right: 40px;
      bottom: 30px;

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
